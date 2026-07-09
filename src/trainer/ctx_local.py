"""
Minimal helpers for the local-context clone (tta_ctx_local, Ours selection).

Rationale (see the appendix on sample selection / a2tta_back.py): in free
delayed-label TTA no persistent sample selection beats adapting on ALL labels —
not even a label-leaking oracle. The one mechanism that consistently helps is to
keep the stable global all-label calibrator and, at each batch, specialise a
*discardable* clone on delayed labels weighted by relevance to the current
context. These two functions provide that weighting; everything else explored is
archived in a2tta_back.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def stack_pool(sel_list, device):
    """List of _Sample (each [N, T]) -> batched pool tensors."""
    x = torch.stack([s.x_flat for s in sel_list]).to(device)      # [P,N,Tin]
    yb = torch.stack([s.y_base for s in sel_list]).to(device)     # [P,N,H]
    y = torch.stack([s.y_flat for s in sel_list]).to(device)      # [P,N,H]
    idx = torch.tensor([float(s.idx) for s in sel_list], device=device)  # [P]
    return {"x": x, "yb": yb, "y": y, "idx": idx, "P": x.shape[0], "N": x.shape[1]}


def _norm_mean1(w, clip=5.0, ess_frac=0.2):
    """Clip to [1/clip,clip], normalise to mean 1; None if ESS collapses (->uniform)."""
    w = w.clamp(min=1.0 / clip, max=clip)
    w = w / w.mean().clamp_min(1e-6)
    ess = (w.sum() ** 2) / (w * w).sum().clamp_min(1e-6)
    if float(ess) < ess_frac * w.numel():
        return None
    return w


def context_row_weights(mode, pool, ctx, calibrator, *, tau=1.0, clip=5.0,
                        ess_frac=0.2, spd=288):
    """Per-row weight [P*N] (window-major, node-minor) mean~1, or None (uniform).

    `mode="hybrid"` weights each delayed sample by relevance to the current
    window: time-of-day / day-of-week phase match, input-window + base-prediction
    cosine similarity, and recency. (`calibrator` is unused here but kept for a
    stable signature.)
    """
    if mode == "uniform":
        return None
    P, N = pool["P"], pool["N"]
    idx = pool["idx"]

    tod_t = float(ctx["target_idx"] % spd); dow_t = float((ctx["target_idx"] // spd) % 7)
    tod = idx % spd
    d = (tod - tod_t).abs(); d = torch.minimum(d, spd - d)
    dow = (idx // spd) % 7
    phase = torch.exp(-d / 24.0) * (0.7 + 0.3 * (dow == dow_t).float())

    fp = torch.cat([pool["x"].mean(1), pool["yb"].mean(1)], dim=-1)        # [P, Tin+H]
    fc = torch.cat([ctx["x_curr"].mean((0, 1)), ctx["yb_curr"].mean((0, 1))], dim=-1)
    fp = F.normalize(fp, dim=-1); fc = F.normalize(fc, dim=-1)
    pat = (fp @ fc).clamp_min(0.0)

    rec = (idx - idx.min()) / (idx.max() - idx.min() + 1e-6)
    ws = phase * (pat + 1e-3) * (0.5 + 0.5 * rec)
    ws = torch.softmax(torch.log(ws.clamp_min(1e-6)) / tau, dim=0) * P
    return _norm_mean1(ws.repeat_interleave(N), clip=clip, ess_frac=ess_frac)
