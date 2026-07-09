"""
Thin wrapper around Salesforce uni2ts Moirai models for the xxltraffic / PEMS
continual-forecasting benchmark. Supports BOTH model families behind one API:

  kind="moirai2" -> Moirai-2.0-R-small  (Moirai2Forecast / Moirai2Module)
                    quantile decoder; point forecast = the 0.5 quantile
                    (== what the GluonTS QuantileForecast.mean falls back to).
  kind="moe"     -> Moirai-MoE base     (MoiraiMoEForecast / MoiraiMoEModule)
                    sample model; point forecast = mean over num_samples draws.

Both expose the SAME tensor-level forward over a batch of univariate context
windows, so we bypass the GluonTS PandasDataset/predictor plumbing (far too slow
for the millions of (window,node) series here) and call `forecast.forward`
directly on tensors we build exactly like `Moirai2Forecast.predict` does.

The wrapper also exposes `loss(ctx, tgt)` for the fine-tune scripts:
  * moirai2: pinball (quantile) loss on the predicted future patch, in the
    model's scaled space — the same quantity the inference path reads out.
  * moe:     PackedNLLLoss on the predictive distribution — the objective the
    shipped uni2ts MoiraiFinetune.training_step uses (uni2ts has no Moirai-MoE
    finetune module, but the loss contract is identical).

Inputs are RAW-scale context (Moirai standardises each series internally via
its packed scaler), matching how the other foundation-model baselines feed data.
"""
import math
import numpy as np


def _impute_rows(x):
    """Per-row (per-series) non-finite -> row mean (0 if all-missing).
    Returns (imputed float32 (N,L), observed bool (N,L))."""
    x = np.asarray(x, dtype=np.float32)
    observed = np.isfinite(x)
    if not observed.all():
        rm = np.nanmean(np.where(observed, x, np.nan), axis=1)
        rm = np.nan_to_num(rm, nan=0.0)
        x = np.where(observed, x, rm[:, None]).astype(np.float32)
    return x, observed


class MoiraiForecaster:
    def __init__(self, kind, repo_id, context_len, horizon_len=12, patch_size=16,
                 num_samples=100, device="cuda", dtype="float32", logger=None):
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

        self.kind = kind
        self.context_len = int(context_len)
        self.horizon_len = int(horizon_len)
        self.device = device
        self.torch = torch
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                      "float16": torch.float16}[dtype]
        self.log = (logger.info if logger else print)

        if kind == "moirai2":
            module = Moirai2Module.from_pretrained(repo_id)
            self.model = Moirai2Forecast(
                module=module, prediction_length=self.horizon_len,
                context_length=self.context_len, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self.patch_size = module.patch_size
            self.quantile_levels = list(module.quantile_levels)
            self.q_median = self.quantile_levels.index(0.5)
        elif kind == "moe":
            module = MoiraiMoEModule.from_pretrained(repo_id)
            self.model = MoiraiMoEForecast(
                module=module, prediction_length=self.horizon_len,
                context_length=self.context_len, target_dim=1,
                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                patch_size=int(patch_size), num_samples=int(num_samples))
            self.patch_size = int(patch_size)
            self.num_samples = int(num_samples)
        else:
            raise ValueError(f"unknown kind {kind!r} (expected moirai2|moe)")

        self.model = self.model.to(device)
        self.module = self.model.module

    # ------------------------------------------------------------------ #
    # input tensors (same construction as Moirai2Forecast.predict)
    # ------------------------------------------------------------------ #
    def _past_tensors(self, ctx):
        """ctx: (B, L) raw. -> past_target (B,L,1) f32, past_observed (B,L,1) bool,
        past_is_pad (B,L) bool. L is assumed == context_len (no padding)."""
        torch = self.torch
        x, observed = _impute_rows(ctx)
        B, L = x.shape
        pt = torch.from_numpy(x[:, :, None]).to(self.device, torch.float32)
        po = torch.from_numpy(observed[:, :, None]).to(self.device, torch.bool)
        pad = torch.zeros((B, L), dtype=torch.bool, device=self.device)
        return pt, po, pad

    # ------------------------------------------------------------------ #
    # zero-shot / inference point forecast
    # ------------------------------------------------------------------ #
    def forecast(self, ctx, horizon=None, chunk_size=1024):
        """ctx: (N, L) raw context. Returns point forecast (N, H)."""
        torch = self.torch
        H = int(horizon or self.horizon_len)
        N = ctx.shape[0]
        self.module.eval()
        out = np.empty((N, H), dtype=np.float32)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            pt, po, pad = self._past_tensors(ctx[s:e])
            with torch.no_grad():
                pred = self.model(past_target=pt, past_observed_target=po,
                                  past_is_pad=pad)            # (b, K, H[, 1])
            if pred.dim() == 4:
                pred = pred.squeeze(-1)
            if self.kind == "moirai2":
                point = pred[:, self.q_median, :]             # 0.5 quantile
            else:
                point = pred.mean(dim=1)                       # mean over samples
            out[s:e] = point[:, :H].float().cpu().numpy()
            if (s // chunk_size) % 20 == 0 or e == N:
                self.log(f"[moirai-{self.kind}]   forecast {e}/{N} series")
        return out

    # ------------------------------------------------------------------ #
    # fine-tune: differentiable loss on (context, target) pairs
    # ------------------------------------------------------------------ #
    def loss(self, ctx, tgt):
        """ctx: (b, L) raw context, tgt: (b, H) raw future. Returns scalar loss
        tensor (graph attached). Train windows are pre-cleaned (finite)."""
        torch = self.torch
        x, observed = _impute_rows(ctx)
        b, L = x.shape
        Ht = tgt.shape[1]
        pt = torch.from_numpy(x[:, :, None]).to(self.device, torch.float32)
        po = torch.from_numpy(observed[:, :, None]).to(self.device, torch.bool)
        pad = torch.zeros((b, L), dtype=torch.bool, device=self.device)
        ft = torch.from_numpy(np.asarray(tgt, np.float32)[:, :, None]).to(self.device, torch.float32)
        fo = torch.ones((b, Ht, 1), dtype=torch.bool, device=self.device)

        (target, observed_mask, sample_id, time_id,
         variate_id, prediction_mask) = self.model._convert(
            self.patch_size, pt, po, pad,
            future_target=ft, future_observed_target=fo)

        if self.kind == "moirai2":
            return self._moirai2_pinball(target, observed_mask, sample_id,
                                         time_id, variate_id, prediction_mask)
        return self._moe_nll(target, observed_mask, sample_id, time_id,
                             variate_id, prediction_mask)

    def _moirai2_pinball(self, target, observed_mask, sample_id, time_id,
                         variate_id, prediction_mask):
        torch = self.torch
        from einops import rearrange
        preds, scaled_target = self.module(
            target, observed_mask, sample_id, time_id, variate_id,
            prediction_mask, training_mode=True)
        Q = self.module.num_quantiles
        P = self.module.patch_size
        NPT = self.module.num_predict_token
        ctl = math.ceil(self.context_len / P)        # context tokens (target_dim=1)
        ptl = math.ceil(self.horizon_len / P)        # prediction tokens
        # preds: (b, seq, NPT*Q*P) -> (b, seq, NPT, Q, P)
        preds = rearrange(preds, "b s (npt q p) -> b s npt q p", npt=NPT, q=Q, p=P)
        # prediction tokens are emitted by the last context token (index ctl-1),
        # first ptl of the NPT look-ahead slots (mirrors structure_multi_predict).
        pred_q = preds[:, ctl - 1, :ptl, :, :]       # (b, ptl, Q, P)
        tgt = scaled_target[:, ctl:ctl + ptl, :]     # (b, ptl, P)
        omask = observed_mask[:, ctl:ctl + ptl, :].to(preds.dtype)  # (b, ptl, P)
        qlev = torch.tensor(self.quantile_levels, device=preds.device,
                            dtype=preds.dtype).view(1, 1, Q, 1)
        err = tgt.unsqueeze(2) - pred_q              # (b, ptl, Q, P)
        pin = torch.maximum(qlev * err, (qlev - 1.0) * err)
        m = omask.unsqueeze(2)                       # (b, ptl, 1, P)
        denom = m.sum() * Q + 1e-8
        return (pin * m).sum() / denom

    def _moe_nll(self, target, observed_mask, sample_id, time_id,
                 variate_id, prediction_mask):
        torch = self.torch
        from uni2ts.loss.packed import PackedNLLLoss
        if not hasattr(self, "_nll"):
            self._nll = PackedNLLLoss()
        patch_size = torch.ones_like(time_id) * self.patch_size
        distr = self.module(target, observed_mask, sample_id, time_id,
                            variate_id, prediction_mask, patch_size)
        return self._nll(pred=distr, target=target, prediction_mask=prediction_mask,
                         observed_mask=observed_mask, sample_id=sample_id,
                         variate_id=variate_id)

    # ------------------------------------------------------------------ #
    def trainable_parameters(self, freeze_encoder=False):
        """Return the parameter list to optimise. freeze_encoder freezes the
        transformer backbone (`module.encoder`) — used for the 0.9B MoE so the
        per-year fine-tune stays cheap; in/out projections + scaler stay trainable."""
        if freeze_encoder and hasattr(self.module, "encoder"):
            for n, p in self.module.named_parameters():
                p.requires_grad = not n.startswith("encoder.")
        else:
            for p in self.module.parameters():
                p.requires_grad = True
        return [p for p in self.module.parameters() if p.requires_grad]
