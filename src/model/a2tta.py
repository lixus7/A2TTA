"""
A2TTA-Lite: Active Adaptive Test-Time Adaptation for Traffic Forecasting.

Components in this file:
  * `ResidualCalibrator` — small per-node MLP that emits a zero-init residual
    delta added on top of a frozen backbone prediction.
  * `ActiveSelector`   — score-based selector over a delayed-label candidate
    pool: weighted blend of recent error / uncertainty / shift / recency.

Both modules are intentionally lightweight (param count ≪ backbone) so that
test-time adaptation only updates a tiny calibrator while the backbone stays
frozen — see `src/trainer/a2tta_trainer.py` for the online loop.
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
    """Lightweight residual calibrator on top of a frozen backbone.

    Forward consumes:
      * y_base  [B*N, T]  — backbone prediction
      * x_in    [B*N, T_in] — input window (z-score normalized in this repo)
      * node_idx[B*N]    — global node ids (0..N_max-1) for embedding lookup

    Output:
      * delta   [B*N, T]  — same shape as `y_base`. Final layer is zero-init
                            so the initial residual is exactly 0 — model
                            behaves identically to the frozen backbone before
                            any warm-up / adaptation.

    Optional MC-dropout: keep `self.training=True` during forward (or call
    .train() on just the calibrator) to get K stochastic forward passes used
    by the active selector's uncertainty term.
    """

    def __init__(
        self,
        num_nodes_max: int,
        x_len: int = 12,
        y_len: int = 12,
        node_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes_max = num_nodes_max
        self.x_len = x_len
        self.y_len = y_len
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim

        # Per-node learnable embedding. Lazily expanded if the graph grows
        # over years (PEMS sensor count grows year over year in this dataset).
        self.node_emb = nn.Parameter(
            torch.zeros(num_nodes_max, node_emb_dim).normal_(0.0, 0.02)
        )

        # Stat features: [last value, mean(x), std(x), slope(x)] = 4 scalars.
        n_stat = 4
        in_dim = y_len + x_len + n_stat + node_emb_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, y_len)

        # Zero-init the output projection so initial residual ≡ 0. This
        # matters: any non-zero init would degrade a well-trained backbone
        # before adaptation has a chance to learn.
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

        # Learnable residual scale, also init small so the residual ramps
        # up gracefully under adaptation.
        self.residual_log_scale = nn.Parameter(torch.tensor(math.log(0.1)))

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
        self.num_nodes_max = new_num_nodes

    @staticmethod
    def temporal_stats(x_in: torch.Tensor) -> torch.Tensor:
        """[B*N, T_in] -> [B*N, 4] = (last, mean, std, slope)."""
        T = x_in.shape[-1]
        last = x_in[..., -1:]
        mean = x_in.mean(dim=-1, keepdim=True)
        std = x_in.std(dim=-1, unbiased=False, keepdim=True)
        # Ordinary least-squares slope vs t = 0..T-1, no extra params.
        t = torch.arange(T, device=x_in.device, dtype=x_in.dtype)
        t = t - t.mean()
        denom = (t * t).sum().clamp_min(1e-6)
        slope = ((x_in - mean) * t).sum(dim=-1, keepdim=True) / denom
        return torch.cat([last, mean, std, slope], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        y_base: torch.Tensor,
        x_in: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """y_pred = y_base + softplus(scale) * delta(...)."""
        stats = self.temporal_stats(x_in)
        emb = self.node_emb[node_idx.clamp_max(self.num_nodes_max - 1)]
        h = torch.cat([y_base, x_in, stats, emb], dim=-1)
        h = self.drop(self.act(self.fc1(h)))
        h = self.drop(self.act(self.fc2(h)))
        delta = self.fc_out(h)
        scale = F.softplus(self.residual_log_scale)
        return y_base + scale * delta

    def predict_with_uncertainty(
        self,
        y_base: torch.Tensor,
        x_in: torch.Tensor,
        node_idx: torch.Tensor,
        K: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K stochastic passes (MC-dropout). Returns (mean, std) of y_pred."""
        was_training = self.training
        # Force dropout to fire while keeping BN-style buffers (none here) intact.
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
    """Score delayed-label candidates and pick top-k for adaptation.

    All inputs are kept on the model's device. The selector itself owns no
    parameters — it just composes precomputed per-sample scores.

    Score = w_err * recent_error
          + w_unc * uncertainty
          + w_shift * shift_score
          + w_recency * recency_score
    """

    def __init__(
        self,
        w_err: float = 1.0,
        w_unc: float = 0.3,
        w_shift: float = 0.3,
        w_recency: float = 0.1,
        mode: str = "active",  # one of: active | random | recent | error_only | all
    ):
        self.w_err = w_err
        self.w_unc = w_unc
        self.w_shift = w_shift
        self.w_recency = w_recency
        assert mode in ("active", "random", "recent", "error_only", "all")
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

        if self.mode == "all":
            return torch.arange(P, device=recent_error.device)
        if self.mode == "random":
            return torch.randperm(P, device=recent_error.device)[:budget]
        if self.mode == "recent":
            return torch.topk(recency, k=budget, largest=True).indices
        if self.mode == "error_only":
            return torch.topk(recent_error, k=budget, largest=True).indices

        score = (
            self.w_err * _safe_norm(recent_error)
            + self.w_unc * _safe_norm(uncertainty)
            + self.w_shift * _safe_norm(shift_score)
            + self.w_recency * _safe_norm(recency)
        )
        return torch.topk(score, k=budget, largest=True).indices
