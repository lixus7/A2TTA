#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 FINE-TUNED (LoRA, per-year retrain) baseline for the evo-xxltraffic
continual-forecasting benchmark — the trained counterpart of the zero-shot
chronos_baseline.py.

Protocol (matches the benchmark's "Retrain" strategy + the other trained
baselines, so it is directly comparable to the main table):
  for each (dataset, year):
    1. take that year's TRAIN split (first 20% of the 31*288 timesteps — the
       same contiguous data the trained ST baselines see).
    2. LoRA-fine-tune a fresh copy of amazon/chronos-2 on it (context_len=12,
       prediction_len=12, to match the eval task exactly).
    3. forecast that year's TEST windows with the fine-tuned model.
    4. score with the identical cumulative-horizon MAE/RMSE/MAPE @3/6/12/Avg.

Everything except the fine-tuning step is reused verbatim from
chronos_baseline.py (window reconstruction, NaN imputation, graph grouping,
uni/multivariate forecasting, metric) so the only difference vs the zero-shot
column is the fine-tuned weights.

Modes: --mode univariate | multivariate (graph-grouped, group_size 64).
Single seed (LoRA fine-tuning has data-sampling randomness; we fix the seed and,
like the zero-shot column, report one run without ±std).

Usage (one (dataset, year, mode) shard — what the katana FIFO calls):
    python chronos_finetune.py --dataset pems03 --year 2001 --mode univariate --gpuid 0

Outputs (under --out-dir, default run_logs/chronos_ft):
    <ds>/<ds>_<year>_<tag>.json   (tag = ft_uni | ft_mv_graph64)
    <ds>/<ds>_results.csv
"""
import os
import sys
import json
import time
import argparse

import numpy as np

# Reuse the zero-shot baseline's verified helpers (data, grouping, metric, fc).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import chronos_baseline as cb
from chronos_baseline import (
    X_LEN, Y_LEN, DAYS, STEPS_PER_DAY,
    build_test_windows, build_groups, impute_windows,
    forecast_univariate, forecast_multivariate, cal_metric,
    resolve_paths, discover_years, append_csv_row,
)


# --------------------------------------------------------------------------- #
# Training data: the per-year TRAIN split (first 20% timesteps), raw scale,
# imputed per series. Mirrors utils/data_convert.generate_samples' train_idx.
# --------------------------------------------------------------------------- #
def build_train_split(raw_data_path, year):
    raw = np.load(os.path.join(raw_data_path, f"{year}.npz"))["x"]
    data = raw[0:DAYS * STEPS_PER_DAY, :]
    t = data.shape[0]
    train = data[0:int(t * 0.2), :]          # (L_train, N) contiguous, raw
    # impute NaN per node-series with that series' mean (all-NaN -> 0)
    x = np.where(np.isfinite(train), train, np.nan).astype(np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    x = np.where(np.isnan(x), mean, x)
    return x.astype(np.float32)              # (L_train, N)


def train_inputs_univariate(train):
    """List of 1-D series, one per node."""
    L, N = train.shape
    return [train[:, n] for n in range(N)]


def train_inputs_multivariate(train, groups):
    """List of 2-D (n_variates, L) series, one per spatial group."""
    return [np.ascontiguousarray(train[:, grp].T) for grp in groups]


# --------------------------------------------------------------------------- #
# Fine-tune
# --------------------------------------------------------------------------- #
def finetune(base_pipeline, inputs, num_steps, lr, batch_size):
    """LoRA-fine-tune a fresh copy; returns the fine-tuned pipeline.
    context_length=X_LEN so training matches the 12->12 eval task exactly."""
    return base_pipeline.fit(
        inputs,
        prediction_length=Y_LEN,
        finetune_mode="lora",
        lora_config=None,           # chronos default LoRA config
        context_length=X_LEN,
        learning_rate=lr,
        num_steps=num_steps,
        batch_size=batch_size,
        disable_data_parallel=True, # one GPU per job (the FIFO gives us 8)
    )


def main():
    ap = argparse.ArgumentParser(description="Chronos-2 LoRA fine-tuned (per-year retrain) baseline")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--begin-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=None)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--mode", default="univariate", choices=["univariate", "multivariate"])
    ap.add_argument("--grouping", default="graph", choices=["graph", "index"])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--gpuid", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    # fine-tune hyper-params
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    ap.add_argument("--ft-batch-size", type=int, default=128, help="series per fine-tune batch")
    ap.add_argument("--fc-batch-size", type=int, default=512, help="series per forecast call")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="run_logs/chronos_ft")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="DEBUG ONLY: cap test windows")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{args.gpuid}"
    else:
        device = "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    paths = resolve_paths(args.dataset)
    if not os.path.isdir(paths["raw"]):
        sys.exit(f"[err] raw data path not found: {paths['raw']}")

    if args.year is not None:
        years = [args.year]
    else:
        years = discover_years(paths["raw"])
        if args.begin_year is not None:
            years = [y for y in years if y >= args.begin_year]
        if args.end_year is not None:
            years = [y for y in years if y <= args.end_year]
    if not years:
        sys.exit("[err] no years to run")

    ds = args.dataset.lower()
    out_dir = os.path.join(args.out_dir, ds)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{ds}_results.csv")
    csv_header = ["dataset", "year", "model", "mode", "grouping", "group_size",
                  "finetune", "num_steps", "lr", "seed", "n_samples", "n_nodes",
                  "n_groups", "T3_MAE", "T3_RMSE", "T3_MAPE", "T6_MAE", "T6_RMSE",
                  "T6_MAPE", "T12_MAE", "T12_RMSE", "T12_MAPE", "Avg_MAE",
                  "Avg_RMSE", "Avg_MAPE", "ft_secs", "secs"]

    print(f"[chronos-ft] {ds} years={years} model={args.model} device={device} "
          f"mode={args.mode} lora steps={args.num_steps} lr={args.lr} "
          f"ft_bs={args.ft_batch_size}", flush=True)

    for year in years:
        t0 = time.time()
        test_x, test_y = build_test_windows(paths["raw"], year, paths["fast"],
                                            verify=not args.no_verify)
        if args.max_windows and test_x.shape[0] > args.max_windows:
            print(f"[chronos-ft] !! DEBUG --max-windows={args.max_windows} "
                  f"(not comparable)", flush=True)
            test_x = test_x[:args.max_windows]; test_y = test_y[:args.max_windows]
        S, _, N = test_x.shape
        gt = np.transpose(test_y, (0, 2, 1))
        test_x, n_bad = impute_windows(test_x)

        # groups (multivariate) — same for train and test, from this year's adj
        groups = None; n_groups = 0
        if args.mode == "multivariate":
            groups = build_groups(args.grouping, paths["graph"], year, N, args.group_size)
            n_groups = len(groups)

        # --- training data for this year ---
        train = build_train_split(paths["raw"], year)
        if args.mode == "multivariate":
            tr_inputs = train_inputs_multivariate(train, groups)
        else:
            tr_inputs = train_inputs_univariate(train)
        print(f"[chronos-ft] {ds} {year}: train_len={train.shape[0]} x {N} nodes; "
              f"test {S} windows; mode={args.mode}"
              + (f" ({n_groups} groups)" if groups else ""), flush=True)

        # --- load base, LoRA fine-tune, forecast ---
        print(f"[chronos-ft] {ds} {year}: loading base + LoRA fine-tune "
              f"({args.num_steps} steps)...", flush=True)
        base = cb.load_pipeline(args.model, device, dtype)
        ft0 = time.time()
        ft = finetune(base, tr_inputs, args.num_steps, args.lr, args.ft_batch_size)
        ft_secs = time.time() - ft0
        print(f"[chronos-ft] {ds} {year}: fine-tuned in {ft_secs:.1f}s; forecasting...",
              flush=True)

        if args.mode == "multivariate":
            pred = forecast_multivariate(ft, test_x, groups, args.fc_batch_size)
        else:
            pred = forecast_univariate(ft, test_x, args.fc_batch_size)

        metrics = cal_metric(gt, pred)
        secs = time.time() - t0
        for T in ("3", "6", "12", "Avg"):
            m = metrics[T]
            print(f"[chronos-ft] {ds} {year} T:{T}\tMAE {m['MAE']:.4f}\t"
                  f"RMSE {m['RMSE']:.4f}\tMAPE {m['MAPE']:.4f}", flush=True)
        print(f"[chronos-ft] {ds} {year} done in {secs:.1f}s (ft {ft_secs:.1f}s)", flush=True)

        meta = {"dataset": ds, "year": year, "model": args.model, "mode": args.mode,
                "grouping": args.grouping if args.mode == "multivariate" else "",
                "group_size": args.group_size if args.mode == "multivariate" else 0,
                "finetune": "lora", "num_steps": args.num_steps, "lr": args.lr,
                "seed": args.seed, "n_samples": int(S), "n_nodes": int(N),
                "n_groups": int(n_groups)}
        rec = dict(meta, ft_secs=round(ft_secs, 1), secs=round(secs, 1), metrics=metrics)
        tag = "ft_uni" if args.mode == "univariate" else f"ft_mv_{args.grouping}{args.group_size}"
        with open(os.path.join(out_dir, f"{ds}_{year}_{tag}_s{args.seed}.json"), "w") as f:
            json.dump(rec, f, indent=2)

        row = dict(meta, ft_secs=round(ft_secs, 1), secs=round(secs, 1))
        for T in ("3", "6", "12", "Avg"):
            pre = "T" + T if T != "Avg" else "Avg"
            row[f"{pre}_MAE"] = round(metrics[T]["MAE"], 4)
            row[f"{pre}_RMSE"] = round(metrics[T]["RMSE"], 4)
            row[f"{pre}_MAPE"] = round(metrics[T]["MAPE"], 4)
        append_csv_row(csv_path, row, csv_header)

        del base, ft
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[chronos-ft] all done. results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
