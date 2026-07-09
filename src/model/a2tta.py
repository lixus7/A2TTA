"""
A2TTA-Lite: Active Adaptive Test-Time Adaptation for Traffic Forecasting.

Components in this file:
  * `ResidualCalibrator` — small per-node calibrator that emits a zero-init
    correction added on top of a frozen backbone prediction. Three architectures
    (`arch=`):
        - "residual" : original MLP residual `y + softplus(s)*delta` (baseline).
        - "norm"     : same MLP but `y_base` is EMA-standardised before the MLP
                       and the residual is predicted in normalised space then
                       scaled back. Fixes the raw-vs-zscore input-scale mismatch.
        - "film"     : per-sample affine calibration `y = gamma*y_base + beta`
                       (FiLM), gamma init 1 / beta init 0 -> identity at start.
  * `ActiveSelector`   — score-based selector over a delayed-label candidate
    pool. Modes: active | random | recent | error_only | low_error | stratified | all.

Both calibrator and selector are intentionally lightweight (param count <<
backbone) — see `src/trainer/a2tta_trainer.py` for the online loop.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Min-max normalize a 1-D tensor to [0, 1]; constant input → zeros."""
    if x.numel() == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < eps:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


class ResidualCalibrator(nn.Module):
    """Lightweight calibrator on top of a frozen backbone.

    Forward consumes:
      * y_base  [B*N, T]    — backbone prediction (RAW scale in this repo)
      * x_in    [B*N, T_in] — input window (z-score normalized in this repo)
      * node_idx[B*N]       — global node ids (0..N_max-1) for embedding lookup

    Output:
      * y_pred  [B*N, T]    — calibrated prediction. All archs are identity at
                              init (zero-init output head) so the model behaves
                              exactly like the frozen backbone before adaptation.

    `arch`:
      * "residual": h = MLP([y_base, x_in, stats, emb]); y = y_base + s*delta.
      * "norm"    : y_base is EMA-standardised (running mean/std over seen y_base)
                    BEFORE the MLP so every feature is ~unit scale; the residual
                    is predicted in normalised space then scaled back by the
                    running std. Fixes fc1 conditioning.
      * "film"    : MLP emits per-sample (gamma, beta); y = gamma*y_base + beta,
                    gamma=1+0.5*tanh(.) (init 1), beta=std*beta_norm (init 0).
    """

    def __init__(
        self,
        num_nodes_max: int,
        x_len: int = 12,
        y_len: int = 12,
        node_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        horizon_emb_dim: int = 0,
        arch: str = "residual",
        rank: int = 8,
        gate_type: str = "channel",
        temporal_kernel: int = 3,
    ):
        super().__init__()
        ARCHS = ("residual", "norm", "film", "affine", "adapter",
                 "lowrank", "gated_adapter", "temporal_conv")
        assert arch in ARCHS, f"unknown arch {arch}; choose from {ARCHS}"
        assert gate_type in ("scalar", "channel")
        self.arch = arch
        self.num_nodes_max = num_nodes_max
        self.x_len = x_len
        self.y_len = y_len
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.horizon_emb_dim = horizon_emb_dim
        self.rank = rank
        self.gate_type = gate_type

        # Per-node learnable embedding. Lazily expanded if the graph grows.
        self.node_emb = nn.Parameter(
            torch.zeros(num_nodes_max, node_emb_dim).normal_(0.0, 0.02)
        )

        # Stat features: [last value, mean(x), std(x), slope(x)] = 4 scalars.
        n_stat = 4
        in_dim = y_len + x_len + n_stat + node_emb_dim
        self.in_dim = in_dim
        self.act = nn.GELU()                       # project default activation
        self.drop = nn.Dropout(p=dropout)

        # --- arch-specific correction head (all identity at init) ---
        # Concat-MLP archs (residual / norm / film): MLP([yb_feat,x,stats,emb]).
        if arch in ("residual", "norm", "film"):
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(hidden_dim, 2 * y_len if arch == "film" else y_len)
            nn.init.zeros_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)
            self.residual_log_scale = nn.Parameter(torch.tensor(math.log(0.1)))
            # Legacy horizon-embedding (residual only); off when horizon_emb_dim==0.
            if horizon_emb_dim > 0:
                self.fc_h = nn.Linear(hidden_dim, horizon_emb_dim)
                self.horizon_emb = nn.Parameter(
                    torch.zeros(y_len, horizon_emb_dim).normal_(0.0, 0.02))

        # Static affine (channel-wise over horizon): y = (1+gamma)*y + beta.
        elif arch == "affine":
            self.affine_gamma = nn.Parameter(torch.zeros(y_len))   # init 0 -> scale 1
            self.affine_beta = nn.Parameter(torch.zeros(y_len))    # init 0 -> shift 0

        # Bottleneck adapter: y = y + std * Wup(GELU(Wdown(concat))); Wup zero-init.
        elif arch == "adapter":
            self.ad_down = nn.Linear(in_dim, rank)
            self.ad_up = nn.Linear(rank, y_len)
            nn.init.zeros_(self.ad_up.weight); nn.init.zeros_(self.ad_up.bias)

        # Low-rank linear residual (no activation): y = y + std * U(V(concat)).
        elif arch == "lowrank":
            self.lr_V = nn.Linear(in_dim, rank, bias=False)
            self.lr_U = nn.Linear(rank, y_len)
            nn.init.zeros_(self.lr_U.weight); nn.init.zeros_(self.lr_U.bias)

        # Gated adapter: y = y + std * gate * Wup(GELU(Wdown(concat))); gate=tanh(g), g=0.
        elif arch == "gated_adapter":
            self.g_down = nn.Linear(in_dim, rank)
            self.g_up = nn.Linear(rank, y_len)
            n_gate = y_len if gate_type == "channel" else 1
            self.gate_g = nn.Parameter(torch.zeros(n_gate))        # tanh(0)=0 -> identity

        # Depthwise temporal conv over the prediction horizon: y = y + DWConv1d(y).
        elif arch == "temporal_conv":
            pad = temporal_kernel // 2
            self.dwconv = nn.Conv1d(1, 1, kernel_size=temporal_kernel, padding=pad)
            nn.init.zeros_(self.dwconv.weight); nn.init.zeros_(self.dwconv.bias)

        # EMA stats of y_base for input standardisation (norm/film archs).
        self.register_buffer("ybase_mean", torch.zeros(1))
        self.register_buffer("ybase_std", torch.ones(1))
        self.register_buffer("ema_inited", torch.zeros(1))
        self.ema_momentum = 0.99

        # Per-node "age" = #times seen across years (for node-age weighting).
        # Persists/grows with the graph alongside node_emb.
        self.register_buffer("node_age", torch.zeros(num_nodes_max))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def expand_nodes(self, new_num_nodes: int):
        """Grow `node_emb` if the current year has more sensors than max-so-far."""
        if new_num_nodes <= self.num_nodes_max:
            return
        extra = new_num_nodes - self.num_nodes_max
        new_rows = torch.zeros(
            extra, self.node_emb_dim, dtype=self.node_emb.dtype, device=self.node_emb.device
        ).normal_(0.0, 0.02)
        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_rows], dim=0))
        self.node_age = torch.cat(
            [self.node_age, torch.zeros(extra, device=self.node_age.device)]
        )
        self.num_nodes_max = new_num_nodes

    @staticmethod
    def temporal_stats(x_in: torch.Tensor) -> torch.Tensor:
        """[B*N, T_in] -> [B*N, 4] = (last, mean, std, slope)."""
        T = x_in.shape[-1]
        last = x_in[..., -1:]
        mean = x_in.mean(dim=-1, keepdim=True)
        std = x_in.std(dim=-1, unbiased=False, keepdim=True)
        t = torch.arange(T, device=x_in.device, dtype=x_in.dtype)
        t = t - t.mean()
        denom = (t * t).sum().clamp_min(1e-6)
        slope = ((x_in - mean) * t).sum(dim=-1, keepdim=True) / denom
        return torch.cat([last, mean, std, slope], dim=-1)

    def _update_ema(self, y_base: torch.Tensor):
        """Update running mean/std of y_base (only while training)."""
        with torch.no_grad():
            m = y_base.mean()
            s = y_base.std(unbiased=False).clamp_min(1e-3)
            if float(self.ema_inited) < 0.5:
                self.ybase_mean.fill_(float(m))
                self.ybase_std.fill_(float(s))
                self.ema_inited.fill_(1.0)
            else:
                mom = self.ema_momentum
                self.ybase_mean.mul_(mom).add_((1 - mom) * m)
                self.ybase_std.mul_(mom).add_((1 - mom) * s)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        y_base: torch.Tensor,
        x_in: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        # ---- archs that act directly on raw y_base (scale-equivariant) ----
        if self.arch == "affine":
            return (1.0 + self.affine_gamma) * y_base + self.affine_beta
        if self.arch == "temporal_conv":
            # depthwise 1-D conv along the prediction-horizon axis of y_base.
            d = self.dwconv(y_base.unsqueeze(1)).squeeze(1)   # [B*N, T] (pad keeps T)
            return y_base + d

        # ---- concat-based archs: features over [yb_feat, x_in, stats, emb] ----
        stats = self.temporal_stats(x_in)
        emb = self.node_emb[node_idx.clamp_max(self.num_nodes_max - 1)]
        if self.arch == "residual":
            yb_feat, sd = y_base, None                 # legacy raw-scale input
        else:
            # standardise y_base so the head's inputs are ~unit scale (norm-fix),
            # then scale the additive correction back by the running std.
            if self.training:
                self._update_ema(y_base)
            mu, sd = self.ybase_mean, self.ybase_std.clamp_min(1e-3)
            yb_feat = (y_base - mu) / sd
        h = torch.cat([yb_feat, x_in, stats, emb], dim=-1)

        if self.arch in ("residual", "norm", "film"):
            h = self.drop(self.act(self.fc1(h)))
            h = self.drop(self.act(self.fc2(h)))
            out = self.fc_out(h)
            if self.arch == "film":
                gamma_raw, beta_norm = out[..., : self.y_len], out[..., self.y_len:]
                gamma = 1.0 + 0.5 * torch.tanh(gamma_raw)     # init 1 (out=0)
                beta = sd * beta_norm                         # init 0
                return gamma * y_base + beta
            delta = out
            if self.arch == "residual" and self.horizon_emb_dim > 0:
                delta = delta + self.fc_h(h) @ self.horizon_emb.T   # legacy term
            scale = F.softplus(self.residual_log_scale)
            if self.arch == "norm":
                delta = sd * delta                            # back to raw units
            return y_base + scale * delta

        if self.arch == "adapter":
            delta = self.ad_up(self.act(self.ad_down(h)))     # ad_up zero-init -> 0
            return y_base + sd * delta
        if self.arch == "lowrank":
            delta = self.lr_U(self.lr_V(h))                   # lr_U zero-init -> 0
            return y_base + sd * delta
        if self.arch == "gated_adapter":
            delta = self.g_up(self.act(self.g_down(h)))
            gate = torch.tanh(self.gate_g)                    # g=0 -> gate 0 -> identity
            return y_base + sd * (gate * delta)
        raise RuntimeError(f"unhandled arch {self.arch}")     # unreachable

    # ------------------------------------------------------------------
    # Parameter report (trainable count + names) — for ablation fairness.
    # ------------------------------------------------------------------
    def param_report(self):
        """Return (total_trainable, head_trainable_excl_node_emb, [(name, shape)])."""
        named = [(n, tuple(p.shape)) for n, p in self.named_parameters() if p.requires_grad]
        total = sum(p.numel() for _, p in
                    ((n, p) for n, p in self.named_parameters() if p.requires_grad))
        head = sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and not n.startswith("node_emb"))
        return total, head, named

    def features(
        self,
        y_base: torch.Tensor,
        x_in: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Penultimate features h (fc2 output, no dropout) — used by the
        gradient-alignment weighting proxy. Shape [*, hidden_dim]."""
        stats = self.temporal_stats(x_in)
        emb = self.node_emb[node_idx.clamp_max(self.num_nodes_max - 1)]
        if self.arch == "residual":
            yb_feat = y_base
        else:
            mu, sd = self.ybase_mean, self.ybase_std.clamp_min(1e-3)
            yb_feat = (y_base - mu) / sd
        h = torch.cat([yb_feat, x_in, stats, emb], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return h

    def head_io(self, y_base, x_in, node_idx):
        """Return (y_pred, h, out): penultimate features `h` and the raw fc_out
        output `out` (kept differentiable) so an oracle can take d(loss)/d(out)
        for an exact, batched last-layer gradient. Mirrors `forward`."""
        stats = self.temporal_stats(x_in)
        emb = self.node_emb[node_idx.clamp_max(self.num_nodes_max - 1)]
        if self.arch == "residual":
            yb_feat = y_base
        else:
            mu, sd = self.ybase_mean, self.ybase_std.clamp_min(1e-3)
            yb_feat = (y_base - mu) / sd
        h = torch.cat([yb_feat, x_in, stats, emb], dim=-1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        out = self.fc_out(h)
        if self.arch == "film":
            gamma = 1.0 + 0.5 * torch.tanh(out[..., : self.y_len])
            beta = self.ybase_std.clamp_min(1e-3) * out[..., self.y_len:]
            y_pred = gamma * y_base + beta
        else:
            delta = out
            if self.arch == "norm":
                delta = self.ybase_std.clamp_min(1e-3) * delta
            y_pred = y_base + F.softplus(self.residual_log_scale) * delta
        return y_pred, h, out

    def predict_with_uncertainty(
        self,
        y_base: torch.Tensor,
        x_in: torch.Tensor,
        node_idx: torch.Tensor,
        K: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K stochastic passes (MC-dropout). Returns (mean, std) of y_pred."""
        was_training = self.training
        self.train(True)
        preds = []
        for _ in range(K):
            preds.append(self.forward(y_base, x_in, node_idx))
        self.train(was_training)
        stack = torch.stack(preds, dim=0)
        return stack.mean(dim=0), stack.std(dim=0, unbiased=False)


# =====================================================================
# Active selector
# =====================================================================
class ActiveSelector:
    """Score delayed-label candidates and pick a subset for adaptation.

    Modes:
      * active     : weighted err+unc+shift+recency, top-k.
      * random     : uniform random subset.
      * recent     : top-k by recency.
      * error_only : top-k by recent error (anomaly-seeking).
      * low_error  : bottom-k by recent error (cleanest samples).
      * stratified : split pool into `budget` recency bins, one per bin
                     (distribution-matched coreset with mild recency).
      * all        : full pool, no selection.
    """

    _MODES = ("active", "random", "recent", "error_only", "low_error", "stratified", "all")

    def __init__(
        self,
        w_err: float = 1.0,
        w_unc: float = 0.3,
        w_shift: float = 0.3,
        w_recency: float = 0.1,
        mode: str = "active",
    ):
        self.w_err = w_err
        self.w_unc = w_unc
        self.w_shift = w_shift
        self.w_recency = w_recency
        assert mode in self._MODES, f"unknown selector mode {mode}"
        self.mode = mode

    def select(
        self,
        recent_error: torch.Tensor,    # [P]
        uncertainty: torch.Tensor,     # [P]
        shift_score: torch.Tensor,     # [P]
        recency: torch.Tensor,         # [P] — larger = more recent
        budget: int,
    ) -> torch.Tensor:
        P = recent_error.shape[0]
        budget = max(1, min(budget, P))
        device = recent_error.device

        if self.mode == "all":
            return torch.arange(P, device=device)
        if self.mode == "random":
            return torch.randperm(P, device=device)[:budget]
        if self.mode == "recent":
            return torch.topk(recency, k=budget, largest=True).indices
        if self.mode == "error_only":
            return torch.topk(recent_error, k=budget, largest=True).indices
        if self.mode == "low_error":
            return torch.topk(recent_error, k=budget, largest=False).indices
        if self.mode == "stratified":
            # Sort by recency, cut into `budget` contiguous bins, take the most
            # recent item of each bin -> spreads the subset across the whole
            # recency range (representative) while leaning slightly recent.
            order = torch.argsort(recency, descending=False)  # oldest..newest
            bins = torch.linspace(0, P, steps=budget + 1, device=device).long()
            picks = []
            for b in range(budget):
                lo, hi = int(bins[b]), int(bins[b + 1])
                if hi <= lo:
                    continue
                picks.append(order[hi - 1])  # newest within the bin
            if not picks:
                return order[-budget:]
            return torch.stack(picks)

        # active
        score = (
            self.w_err * _safe_norm(recent_error)
            + self.w_unc * _safe_norm(uncertainty)
            + self.w_shift * _safe_norm(shift_score)
            + self.w_recency * _safe_norm(recency)
        )
        return torch.topk(score, k=budget, largest=True).indices
