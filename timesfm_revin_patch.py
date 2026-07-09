"""Mask-aware RevIN patch for HuggingFace TimesFM-2.5 (TimesFm2_5ModelForPrediction).

Why: the stock forward computes the RevIN normalization stats over the FULL
(padded) context: `mu = input_ts.mean(1); sigma = input_ts.std(1)`. When we feed
a SHORT real context (e.g. 12 steps) front-padded to a 32-multiple, those zero
pads corrupt mu/sigma. This patch computes mu/sigma over the NON-padded
positions only (using the model's own `input_padding` mask).

Backward-compatible: when there is no padding (the original 32-real-step path),
the masked stats equal the original `mean`/unbiased-`std`, so existing 32-step
runs are numerically unchanged. This lets TimesFM-FT run at a true 12-step
lookback (matching the baselines / zero-shot column), with correct per-window
normalization.

Apply once at process start:  import timesfm_revin_patch; timesfm_revin_patch.apply()
"""
import torch
import torch.nn.functional as F


def _patched_forward(
    self,
    past_values,
    window_size=None,
    future_values=None,
    forecast_context_len=None,
    truncate_negative=None,
    force_flip_invariance=None,
    **kwargs,
):
    from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5OutputForPrediction

    forecast_context_len = forecast_context_len or self.context_len
    device = past_values[0].device

    inputs = [ts[-forecast_context_len:] for ts in past_values]
    input_min = torch.min(torch.stack([torch.min(ts) for ts in inputs]))

    if window_size is not None:
        new_inputs = []
        for ts in inputs:
            new_inputs.extend(self._timesfm_moving_average(ts, window_size))
        inputs = new_inputs

    if truncate_negative is None:
        truncate_negative = self.config.infer_is_positive
    if force_flip_invariance is None:
        force_flip_invariance = self.config.force_flip_invariance

    input_ts, input_padding = self._preprocess(inputs, context_len=forecast_context_len)
    input_ts = input_ts.to(device)
    input_padding = input_padding.to(device)

    # ---- MASK-AWARE RevIN stats (the only change vs the stock forward) -------
    # input_padding: 1 = padded position, 0 = real. Restrict to the context span.
    ctx_pad = input_padding[:, : input_ts.shape[1]]
    real = (1.0 - ctx_pad).to(input_ts.dtype)                 # 1 = real, 0 = pad
    cnt = real.sum(dim=1, keepdim=True)                       # # real steps / row
    mu_global = (input_ts * real).sum(dim=1, keepdim=True) / cnt.clamp(min=1.0)
    var = ((input_ts - mu_global) ** 2 * real).sum(dim=1, keepdim=True) / (cnt - 1.0).clamp(min=1.0)
    sigma_global = torch.sqrt(var)
    # When real==all-ones (no padding): mu==mean, var==unbiased var -> identical
    # to the stock input_ts.mean(1)/input_ts.std(1).
    # --------------------------------------------------------------------------

    normalized_ts = self.model._revin(input_ts, mu_global, sigma_global, reverse=False)

    pf_outputs, quantile_spreads, model_outputs = self._decode_and_project(normalized_ts, input_padding, **kwargs)

    if force_flip_invariance:
        flipped_pf, flipped_qs, _ = self._decode_and_project(-normalized_ts, input_padding, **kwargs)

        def _flip_quantiles(x):
            return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1,))], dim=-1)

        pf_outputs = (pf_outputs - _flip_quantiles(flipped_pf)) / 2
        quantile_spreads = (quantile_spreads - _flip_quantiles(flipped_qs)) / 2

    horizon = min(self.horizon_len, pf_outputs.shape[1])
    full_forecast = pf_outputs[:, :horizon, :].clone()

    median_index = min(self.config.decode_index, full_forecast.shape[-1] - 1)
    if self.config.use_continuous_quantile_head:
        max_quantile_horizon = min(horizon, quantile_spreads.shape[1])
        for idx, _ in enumerate(self.config.quantiles, start=1):
            if idx == median_index or idx >= full_forecast.shape[-1]:
                continue
            full_forecast[:, :max_quantile_horizon, idx] = (
                quantile_spreads[:, :max_quantile_horizon, idx]
                - quantile_spreads[:, :max_quantile_horizon, median_index]
                + full_forecast[:, :max_quantile_horizon, median_index]
            )

    full_predictions = self.model._revin(full_forecast, mu_global, sigma_global, reverse=True)
    decode_index = min(self.config.decode_index, full_predictions.shape[-1] - 1)
    mean_predictions = full_predictions[:, :, decode_index]

    if window_size is not None:
        mean_predictions = mean_predictions[0::2, ...] + mean_predictions[1::2, ...]
        full_predictions = full_predictions[0::2, ...] + full_predictions[1::2, ...]

    if truncate_negative:
        zero = torch.zeros(1, device=mean_predictions.device, dtype=mean_predictions.dtype)
        clamped_mean = torch.maximum(mean_predictions, zero)
        clamped_full = torch.maximum(full_predictions, zero)
        should_clamp = (input_min >= 0).to(mean_predictions.device)
        mean_predictions = torch.where(should_clamp, clamped_mean, mean_predictions)
        full_predictions = torch.where(should_clamp, clamped_full, full_predictions)

    loss = None
    if future_values is not None:
        target_len = future_values.shape[1]
        normalized_preds = full_forecast[:, :target_len]
        normalized_targets = self.model._revin(future_values, mu_global, sigma_global, reverse=False)
        normalized_mean_preds = normalized_preds[:, :, decode_index]
        mse_loss = F.mse_loss(normalized_mean_preds, normalized_targets)
        quantile_indices = [i for i in range(normalized_preds.shape[-1]) if i != decode_index]
        if quantile_indices:
            index_tensor = torch.tensor(quantile_indices, device=normalized_preds.device, dtype=torch.long)
            quantile_tensor = torch.index_select(normalized_preds, dim=-1, index=index_tensor)
            quantile_loss = self._quantile_loss(quantile_tensor, normalized_targets)
            loss = mse_loss + quantile_loss
        else:
            loss = mse_loss

    return TimesFm2_5OutputForPrediction(
        last_hidden_state=model_outputs.last_hidden_state,
        hidden_states=model_outputs.hidden_states,
        attentions=model_outputs.attentions,
        mean_predictions=mean_predictions,
        full_predictions=full_predictions,
        loss=loss,
    )


def apply(logger=None):
    from transformers.models.timesfm2_5 import modeling_timesfm2_5 as M
    M.TimesFm2_5ModelForPrediction.forward = _patched_forward
    msg = "[timesfm_revin_patch] mask-aware RevIN applied to TimesFm2_5ModelForPrediction.forward"
    (logger.info if logger is not None else print)(msg)
