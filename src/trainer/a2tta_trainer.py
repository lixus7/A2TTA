"""
A2TTA-Lite trainer & online evaluator.

Pipeline per year:
  1. Load frozen backbone checkpoint (Online-AN by default; configurable).
  2. (Optional) warm up the residual calibrator on the year's train split,
     keeping the backbone frozen. L1 loss on raw-scale targets.
  3. Run causal online evaluation on the test split:
        * test windows are reordered to true chronological order
          (`generate_dataset` produces them latest-first, see data_convert.py)
        * each window enters a `pending_queue`; once its target horizon has
          elapsed, it is released into `candidate_pool`
        * every `adapt_every_batches` we (a) score candidates with
          ActiveSelector, (b) take `adapt_steps` gradient steps on the
          calibrator only (backbone stays frozen)
  4. Compute per-horizon metrics with the same `cal_metric` used by every
     other baseline so numbers are directly comparable to Online-AN etc.

Anti-leakage invariant: a sample's y is never read until its release time.
The `pending_queue` stores y but the adaptation code only ever pops from
`candidate_pool` (i.e. post-release) entries.
"""
from __future__ import annotations

import os
import csv
import copy
import math
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from src.model.a2tta import ResidualCalibrator, ActiveSelector
# Minimal helpers for the local-context clone (tta_ctx_local, Ours selection).
from src.trainer.ctx_local import stack_pool, context_row_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(pred_flat, y_flat, batch):
    pred, _ = to_dense_batch(pred_flat, batch=batch)
    y, _ = to_dense_batch(y_flat, batch=batch)
    return pred, y


def _build_node_idx(B: int, N: int, device) -> torch.Tensor:
    """[B*N] tensor of node ids 0..N-1 repeated for each batch element."""
    base = torch.arange(N, device=device)
    return base.unsqueeze(0).expand(B, N).reshape(-1)


class _XOnly:
    """Minimal stand-in for a PyG batch when a model's forward only reads .x."""
    __slots__ = ("x",)
    def __init__(self, x):
        self.x = x


def _backbone_infer(backbone, data, adj, N, chunk: int = 16):
    """Inference-only backbone call. For STAEFORMER the spatial-attention
    activations scale with batch×N and OOM small GPUs (L40S died at year 2011
    on PEMS03 with eval bs=64) — so chunk the batch internally; the logical
    batch (and thus TTA cadence) is unchanged. Legacy backbones pass through
    untouched. Call under torch.no_grad()."""
    if not hasattr(getattr(backbone, "backbone", None), "adaptive_embedding"):
        return backbone(data, adj)
    x = data.x
    B = x.shape[0] // N
    if B <= chunk:
        return backbone(data, adj)
    outs = []
    for i in range(0, B, chunk):
        outs.append(backbone(_XOnly(x[i * N:(i + chunk) * N]), adj))
    return torch.cat(outs, dim=0)


def _staeformer_forward_flat(model, x3: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Grad-enabled STAEFORMER_Model forward from a dense [B, N, T_in] tensor.

    Mirrors STAEFORMER_Model.forward but skips the PyG Data wrapper (and its
    eval-mode no-op dropout). Used by the a2tta_emb mode, which needs a FRESH
    backbone forward so gradients reach the adaptive embedding — the pool's
    cached y_base tensors are detached and cannot carry gradients.
    """
    T = x3.shape[-1]
    feat = model.backbone(x3, adj).reshape(-1, T)
    h = feat + x3.reshape(-1, T)
    return model.fc(model.activation(h))


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warmup_calibrator(
    backbone: nn.Module,
    calibrator: ResidualCalibrator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    epochs: int = 5,
    lr: float = 1e-3,
):
    """Train calibrator on train split, backbone frozen."""
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    opt = optim.AdamW(filter(lambda p: p.requires_grad, calibrator.parameters()), lr=lr)
    N = args.adj.shape[0]
    infer_chunk = int(getattr(args, "a2tta", {}).get("infer_chunk", 16))

    best_val = float("inf")
    for ep in range(epochs):
        calibrator.train()
        total, cnt = 0.0, 0
        for data in train_loader:
            data = data.to(args.device, non_blocking=True)
            with torch.no_grad():
                y_base = _backbone_infer(backbone, data, args.adj, N, infer_chunk)  # [B*N, T]
            B = y_base.shape[0] // N
            x_in = data.x.reshape(-1, N, args.gcn["in_channel"]).reshape(-1, args.gcn["in_channel"])
            node_idx = _build_node_idx(B, N, args.device)
            y_pred = calibrator(y_base, x_in, node_idx)
            loss = F.l1_loss(y_pred, data.y, reduction="mean")
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss); cnt += 1

        # Validation
        calibrator.eval()
        v_total, v_cnt = 0.0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(args.device, non_blocking=True)
                y_base = _backbone_infer(backbone, data, args.adj, N, infer_chunk)
                B = y_base.shape[0] // N
                x_in = data.x.reshape(-1, N, args.gcn["in_channel"]).reshape(-1, args.gcn["in_channel"])
                node_idx = _build_node_idx(B, N, args.device)
                y_pred = calibrator(y_base, x_in, node_idx)
                loss = masked_mae_np(data.y.cpu().numpy(), y_pred.cpu().numpy(), 0)
                if not np.isfinite(loss):
                    continue
                v_total += float(loss); v_cnt += 1
        v_mae = v_total / max(v_cnt, 1)
        if hasattr(args, "logger"):
            tr = total / max(cnt, 1)
            args.logger.info(
                f"  [warmup] year={args.year} ep={ep} train_l1={tr:.4f} val_mae={v_mae:.4f}"
            )
        if v_mae < best_val:
            best_val = v_mae
    return best_val


# ---------------------------------------------------------------------------
# Online delayed-label evaluation with active selection
# ---------------------------------------------------------------------------

class _Sample:
    """One chronological window kept in queues for delayed-label adaptation."""
    __slots__ = ("idx", "x_flat", "y_flat", "node_idx", "y_base", "y_pred", "release_idx")

    def __init__(self, idx, x_flat, y_flat, node_idx, y_base, y_pred, release_idx):
        self.idx = idx
        self.x_flat = x_flat
        self.y_flat = y_flat
        self.node_idx = node_idx
        self.y_base = y_base
        self.y_pred = y_pred
        self.release_idx = release_idx


def _chronological_test_arrays(inputs):
    """`generate_dataset` emits test windows latest-first; reverse to chronological."""
    test_x = inputs["test_x"][::-1].copy()
    test_y = inputs["test_y"][::-1].copy()
    return test_x, test_y


def _shift_score_against_source(
    x_in_flat: torch.Tensor,
    src_mean: torch.Tensor,
    src_std: torch.Tensor,
) -> torch.Tensor:
    """Per-sample distance between input window stats and source train stats.

    Both inputs already in z-score space; we use mean+slope of the window.
    Returns [B*N] -> averaged later to per-sample [P].
    """
    m = x_in_flat.mean(dim=-1)
    s = x_in_flat.std(dim=-1, unbiased=False)
    z_m = (m - src_mean[0]) / (src_std[0] + 1e-6)
    z_s = (s - src_mean[1]) / (src_std[1] + 1e-6)
    return (z_m.abs() + z_s.abs())  # [B*N]


def online_a2tta_eval(
    backbone: nn.Module,
    calibrator: ResidualCalibrator,
    inputs,
    args,
):
    """Run causal online TTA on the year's test split. Logs per-horizon MAE/RMSE/MAPE
    via cal_metric, identical to test_model so numbers are directly comparable.
    """
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    N = args.adj.shape[0]
    H = args.y_len
    device = args.device

    cfg = args.a2tta
    method_mode: str = cfg["method"]   # "backbone" | "calibrator" | "tta_random" | "tta_recent" | "tta_error" | "a2tta_lite" | "tta_all"
    adapt_lr: float = cfg["adapt_lr"]
    adapt_steps: int = cfg["adapt_steps"]
    adapt_every: int = cfg["adapt_every_batches"]
    cand_pool_size: int = cfg["candidate_pool_size"]
    budget_frac: float = cfg["budget_frac"]
    lambda_cons: float = cfg["lambda_cons"]
    lambda_reg: float = cfg["lambda_reg"]
    fast_dev_run: bool = cfg.get("fast_dev_run", False)
    bs: int = int(cfg.get("eval_batch_size", 64))
    infer_chunk: int = int(cfg.get("infer_chunk", 16))
    mc_K: int = int(cfg.get("mc_K", 4))

    # Adaptation enabled?
    do_adapt = method_mode in (
        "tta_random", "tta_recent", "tta_error", "a2tta_lite", "tta_all",
        "a2tta_emb", "tta_ctx_local",
    )
    use_calibrator = method_mode != "backbone"

    if method_mode == "tta_random":
        selector = ActiveSelector(mode="random")
    elif method_mode == "tta_recent":
        selector = ActiveSelector(mode="recent")
    elif method_mode == "tta_error":
        selector = ActiveSelector(mode="error_only")
    elif method_mode in ("tta_all", "tta_ctx_local"):
        # Full-pool adaptation; bypasses the active scoring pass. tta_ctx_local
        # updates the global calibrator exactly like tta_all, then specialises a
        # discardable local clone at prediction time (see the predict block).
        selector = ActiveSelector(mode="all")
    else:
        selector = ActiveSelector(
            w_err=cfg.get("w_err", 1.0), w_unc=cfg.get("w_unc", 0.3),
            w_shift=cfg.get("w_shift", 0.3), w_recency=cfg.get("w_recency", 0.1),
            mode="active",
        )

    # tta_ctx_local (Ours selection): keep the stable global all-label calibrator,
    # then at each batch clone it and take a few steps on context-weighted delayed
    # labels before predicting (the clone is discarded). The exploration of other
    # selection / weighting ideas that did NOT beat tta_all is archived in
    # a2tta_back.py — see appendix on sample selection.
    ctx_local = method_mode == "tta_ctx_local"
    local_steps = int(cfg.get("local_steps", 3))

    # Snapshot the calibrator init for proximal regularizer.
    init_state = {k: v.detach().clone() for k, v in calibrator.state_dict().items()}

    # Optimizer for the (small) calibrator.
    opt = None
    if do_adapt:
        opt = optim.AdamW(calibrator.parameters(), lr=adapt_lr)

    # a2tta_emb: additionally adapt the backbone's spatio-temporal adaptive
    # embedding (STAEFormer's per-(timestep,node) table) — but ONLY rows of
    # actively-selected nodes, chosen by per-node delayed-label error. The
    # rest of the backbone stays frozen. All other method modes leave the
    # backbone untouched (emb_param stays None), so legacy paths are intact.
    emb_param = None
    emb_init = None
    emb_opt = None
    if method_mode == "a2tta_emb":
        emb_param = getattr(getattr(backbone, "backbone", None), "adaptive_embedding", None)
        if emb_param is None:
            raise ValueError(
                "a2tta_emb requires a STAEFORMER backbone "
                "(backbone.backbone.adaptive_embedding not found) — pass "
                "--backbone_method STAEFORMER with matching checkpoints."
            )
        emb_param.requires_grad_(True)
        emb_init = emb_param.detach().clone()
        # weight_decay=0: rows whose grad we zero must not drift via decay.
        emb_opt = optim.AdamW([emb_param], lr=float(cfg.get("emb_lr", adapt_lr)), weight_decay=0.0)
    node_budget_frac = float(cfg.get("node_budget_frac", 0.25))
    emb_batch_cap = int(cfg.get("emb_batch_cap", 16))
    lambda_reg_emb = float(cfg.get("lambda_reg_emb", lambda_reg))

    # Source-stats reference for shift_score (use train split mean/std of last+slope features).
    train_x = inputs["train_x"]  # [T, x_len, N] (z-score already)
    src_m = float(np.mean(train_x))
    src_s = float(np.std(train_x))
    # use mean of mean / mean of std as crude reference
    src_mean = torch.tensor([src_m, np.std(train_x.reshape(-1, train_x.shape[-1]).mean(axis=0))],
                            device=device, dtype=torch.float32)
    src_std = torch.tensor([np.std(train_x), 1.0], device=device, dtype=torch.float32)

    # Reverse test arrays into chronological order, then build a vanilla DataLoader.
    test_x, test_y = _chronological_test_arrays(inputs)
    chrono_inputs = {"test_x": test_x, "test_y": test_y}
    test_loader = DataLoader(
        SpatioTemporalDataset(chrono_inputs, "test"),
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=0,  # ordering matters; keep determinism
    )

    pending = deque()           # _Sample objects, ordered by chronological idx
    pool: deque[_Sample] = deque(maxlen=cand_pool_size)

    pred_chrono, truth_chrono = [], []
    next_idx = 0    # global chronological index of the next window to be predicted
    batch_idx = 0
    n_adapt_steps_total = 0

    for batch in test_loader:
        if fast_dev_run and batch_idx >= 4:
            break

        batch = batch.to(device, non_blocking=True)
        # `to_dense_batch` is correct for variable-N sparse Data, but our
        # SpatioTemporalDataset already yields fixed [N, T] per item, so each
        # sample contributes exactly N rows. Recover B from the data.
        B = batch.x.shape[0] // N
        x_flat = batch.x       # [B*N, T_in]
        y_flat = batch.y       # [B*N, T]
        node_idx = _build_node_idx(B, N, device)

        # ---------- Release samples from pending → candidate pool ----------
        # A sample with global idx i is released once next_idx >= i + H.
        while pending and pending[0].release_idx <= next_idx:
            pool.append(pending.popleft())

        # ---------- Adapt calibrator on current pool ----------
        if do_adapt and len(pool) >= max(8, int(0.1 * cand_pool_size)) and (batch_idx % adapt_every == 0):
            n_steps = adapt_steps
            for _ in range(n_steps):
                # Build per-sample features for selection (cheap, on existing tensors).
                ps = list(pool)
                P = len(ps)

                # recent_error (per-sample MAE from cached predictions vs delayed y)
                with torch.no_grad():
                    rec_err = torch.stack([
                        (s.y_pred - s.y_flat).abs().mean() for s in ps
                    ])
                    # uncertainty via MC-dropout on calibrator only
                    if mc_K > 1 and selector.mode == "active":
                        unc_list = []
                        for s in ps:
                            _, std = calibrator.predict_with_uncertainty(
                                s.y_base, s.x_flat, s.node_idx, K=mc_K
                            )
                            unc_list.append(std.mean())
                        unc = torch.stack(unc_list)
                    else:
                        unc = torch.zeros(P, device=device)

                    if selector.mode == "active":
                        shift = torch.stack([
                            _shift_score_against_source(s.x_flat, src_mean, src_std).mean()
                            for s in ps
                        ])
                    else:
                        shift = torch.zeros(P, device=device)

                    # Recency: bigger idx = more recent. Normalize via _safe_norm later.
                    recency = torch.tensor([float(s.idx) for s in ps], device=device)

                budget = max(1, int(budget_frac * P))
                sel = selector.select(rec_err, unc, shift, recency, budget=budget)
                sel_list = [ps[int(i)] for i in sel.tolist()]

                # Stack selected samples into a single big batch for one update.
                xs = torch.cat([s.x_flat for s in sel_list], dim=0)
                ys = torch.cat([s.y_flat for s in sel_list], dim=0)
                ybs = torch.cat([s.y_base for s in sel_list], dim=0)
                ns = torch.cat([s.node_idx for s in sel_list], dim=0)

                calibrator.train()
                y_hat = calibrator(ybs, xs, ns)
                loss_sup = F.l1_loss(y_hat, ys, reduction="mean")

                # Optional weak consistency loss (cheap noise aug on x).
                if lambda_cons > 0:
                    noise1 = 0.01 * torch.randn_like(xs)
                    noise2 = 0.01 * torch.randn_like(xs)
                    y1 = calibrator(ybs, xs + noise1, ns)
                    y2 = calibrator(ybs, xs + noise2, ns)
                    loss_cons = F.l1_loss(y1, y2, reduction="mean")
                else:
                    loss_cons = torch.zeros((), device=device)

                # Proximal regularizer to anchor near init (avoid overfitting to small pool).
                if lambda_reg > 0:
                    loss_reg = torch.zeros((), device=device)
                    for k, p in calibrator.named_parameters():
                        if k in init_state:
                            loss_reg = loss_reg + ((p - init_state[k]) ** 2).sum()
                else:
                    loss_reg = torch.zeros((), device=device)

                loss = loss_sup + lambda_cons * loss_cons + lambda_reg * loss_reg
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                n_adapt_steps_total += 1

                # ---- a2tta_emb: update actively-selected adaptive-emb rows ----
                if emb_param is not None:
                    # Active NODE selection: rank nodes by mean delayed-label
                    # error over the selected samples (extends the sample-level
                    # active principle to the node axis).
                    with torch.no_grad():
                        err_node = torch.stack([
                            (s.y_pred - s.y_flat).abs().mean(dim=-1) for s in sel_list
                        ]).mean(dim=0)                                   # [N]
                        k = max(1, int(node_budget_frac * err_node.shape[0]))
                        node_mask = torch.zeros_like(err_node, dtype=torch.bool)
                        node_mask[torch.topk(err_node, k=k).indices] = True
                    # Fresh backbone forward on a capped sub-batch (gradients
                    # cannot flow through the pool's cached y_base).
                    sub = sel_list[:emb_batch_cap]
                    Bs = len(sub)
                    xs_e = torch.cat([s.x_flat for s in sub], dim=0)
                    ys_e = torch.cat([s.y_flat for s in sub], dim=0)
                    ns_e = torch.cat([s.node_idx for s in sub], dim=0)
                    y_base_fresh = _staeformer_forward_flat(
                        backbone, xs_e.reshape(Bs, N, -1), args.adj)
                    y_hat_e = calibrator(y_base_fresh, xs_e, ns_e)
                    loss_e = (F.l1_loss(y_hat_e, ys_e, reduction="mean")
                              + lambda_reg_emb * ((emb_param - emb_init) ** 2).sum())
                    if torch.isfinite(loss_e):
                        emb_opt.zero_grad(set_to_none=True)
                        # Calibrator grads from this backward are discarded by
                        # its own zero_grad at the next calibrator step.
                        loss_e.backward()
                        if emb_param.grad is not None:
                            emb_param.grad[:, ~node_mask, :] = 0
                        emb_opt.step()

        # ---------- Predict current batch (post-adapt calibrator) ----------
        calibrator.eval()
        with torch.no_grad():
            y_base = _backbone_infer(backbone, batch, args.adj, N, infer_chunk)             # [B*N, T]

        if ctx_local and len(pool) >= max(8, int(0.1 * cand_pool_size)):
            # Ours (selection): clone the stable global calibrator, specialise the
            # clone on context-weighted delayed labels for a few steps, predict
            # with it, then discard it — the global calibrator is never biased.
            local = copy.deepcopy(calibrator); local.train()
            lopt = optim.AdamW(local.parameters(), lr=adapt_lr)
            pool_d = stack_pool(list(pool), device)
            ctx_l = {"x_curr": x_flat.reshape(B, N, -1), "yb_curr": y_base.reshape(B, N, -1),
                     "target_idx": next_idx}
            xs_l = pool_d["x"].reshape(-1, pool_d["x"].shape[-1])
            ys_l = pool_d["y"].reshape(-1, pool_d["y"].shape[-1])
            ybs_l = pool_d["yb"].reshape(-1, pool_d["yb"].shape[-1])
            ns_l = torch.arange(N, device=device).repeat(pool_d["P"])
            for _ in range(local_steps):
                rw = context_row_weights("hybrid", pool_d, ctx_l, local)
                yh = local(ybs_l, xs_l, ns_l)
                pr = (yh - ys_l).abs().mean(dim=-1)
                loss = (pr * rw).mean() if rw is not None else pr.mean()
                if not torch.isfinite(loss):
                    break
                lopt.zero_grad(set_to_none=True); loss.backward(); lopt.step()
            local.eval()
            with torch.no_grad():
                y_pred_flat = local(y_base, x_flat, node_idx)
            del local, lopt
        else:
            with torch.no_grad():
                if use_calibrator:
                    y_pred_flat = calibrator(y_base, x_flat, node_idx)
                else:
                    y_pred_flat = y_base

        pred_dense, y_dense = _to_dense(y_pred_flat, y_flat, batch.batch)
        pred_chrono.append(pred_dense.detach().cpu().numpy())
        truth_chrono.append(y_dense.detach().cpu().numpy())

        # ---------- Enqueue this batch's samples for future delayed adaptation ----------
        # Slice the flat tensors back into per-sample chunks of N rows.
        for j in range(B):
            i_global = next_idx + j
            x_j = x_flat[j * N : (j + 1) * N].detach()
            y_j = y_flat[j * N : (j + 1) * N].detach()
            yb_j = y_base[j * N : (j + 1) * N].detach()
            yp_j = y_pred_flat[j * N : (j + 1) * N].detach()
            n_j = node_idx[j * N : (j + 1) * N].detach()
            pending.append(_Sample(
                idx=i_global, x_flat=x_j, y_flat=y_j, node_idx=n_j,
                y_base=yb_j, y_pred=yp_j, release_idx=i_global + H,
            ))
        next_idx += B
        batch_idx += 1

    # Stack predictions in chronological order, run cal_metric.
    pred_arr = np.concatenate(pred_chrono, axis=0)
    truth_arr = np.concatenate(truth_chrono, axis=0)
    cal_metric(truth_arr, pred_arr, args)

    if hasattr(args, "logger"):
        args.logger.info(
            f"  [a2tta] year={args.year} method={method_mode} "
            f"adapt_steps={n_adapt_steps_total} pool_size_final={len(pool)} "
            f"pending_final={len(pending)}"
        )
    return n_adapt_steps_total


# ---------------------------------------------------------------------------
# CSV result writer (one row per year × method × seed × horizon)
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "dataset", "method", "seed", "year", "horizon",
    "MAE", "RMSE", "MAPE",
    "adapt_lr", "adapt_steps", "budget_frac", "candidate_pool_size",
    "lambda_cons", "hidden_dim",
]


def append_results_csv(args, csv_path: str):
    """Pull args.result and append per-year per-horizon rows."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        cfg = args.a2tta
        for h in ["3", "6", "12", "Avg"]:
            res = args.result.get(h, {})
            mae_d = res.get(" MAE", {})
            rmse_d = res.get("RMSE", {})
            mape_d = res.get("MAPE", {})
            for year in sorted(set(mae_d.keys()) | set(rmse_d.keys()) | set(mape_d.keys())):
                w.writerow({
                    "dataset": cfg.get("dataset", "PEMS05"),
                    "method": cfg.get("method", "?"),
                    "seed": args.seed,
                    "year": year,
                    "horizon": h,
                    "MAE": mae_d.get(year, ""),
                    "RMSE": rmse_d.get(year, ""),
                    "MAPE": mape_d.get(year, ""),
                    "adapt_lr": cfg.get("adapt_lr", ""),
                    "adapt_steps": cfg.get("adapt_steps", ""),
                    "budget_frac": cfg.get("budget_frac", ""),
                    "candidate_pool_size": cfg.get("candidate_pool_size", ""),
                    "lambda_cons": cfg.get("lambda_cons", ""),
                    "hidden_dim": cfg.get("hidden_dim", ""),
                })
