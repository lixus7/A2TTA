#!/usr/bin/env python3
"""
evaluate_new_sensors_a2tta.py

Re-run **A2TTA-Lite** (Ours) online test-time adaptation per year and report
MAE / RMSE / MAPE restricted to the **newly added sensors** (node indices
[prev_year_N, cur_year_N)). Output mirrors scripts/evaluate_new_sensors.py so
make_new_sensor_tables.py can merge it like any other baseline.

Why a separate script: A2TTA produces its predictions *online* (causal, with
delayed-label active adaptation), so unlike the static baselines we cannot just
reload a frozen checkpoint and forward once. We re-run the exact same online
eval that produced the main-table A2TTA numbers (hyperparameters mirror
scripts/a2tta_lite_all_datasets_6gpu.sh) and slice the chronological
prediction/truth arrays to the new-sensor columns.

The trick: a2tta_trainer.online_a2tta_eval() calls cal_metric(truth, pred, args)
exactly once per year with the full [windows, N, T] arrays. We monkeypatch that
single call to (a) recompute the all-sensor metrics (sanity, == main table) and
(b) compute the new-sensor-slice metrics with the *same* masked-metric math as
the other baselines (scripts.evaluate_new_sensors._slice_metrics).

Per-job output: log/<DS>/a2tta_lite_<lower>-<seed>/eval_new_sensors.{csv,log}
  csv columns: year,n_total,n_new,scope,horizon,MAE,RMSE,MAPE

Usage (run from the eac/ directory):
  python scripts/evaluate_new_sensors_a2tta.py \
      --conf conf/PEMS03/a2tta_lite_pems03.json --seed 51 \
      --dataset PEMS03 --gpuid 0
"""
import argparse
import csv
import os.path as osp
import sys

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, "src"))

import numpy as np
import networkx as nx
import torch

from utils.initialize import init, seed_anything, init_log
import a2tta_main as A
import src.trainer.a2tta_trainer as a2t
from src.model.a2tta import ResidualCalibrator
from scripts.evaluate_new_sensors import _slice_metrics


# Tuned Ours hyperparameters. DEFAULT method = tta_ctx_local (full Ours = the
# main-table A2TTA), NOT the legacy a2tta_lite. Override via --method if needed.
A2TTA_HP = dict(
    method="tta_ctx_local",
    backbone_method="TrafficStream",
    freeze_backbone=1,
    adapter_hidden_dim=64,
    node_emb_dim=16,
    calibrator_dropout=0.1,
    adapt_lr=1e-3,
    adapt_steps=3,
    adapt_every_batches=1,
    budget_frac=0.25,
    candidate_pool_size=512,
    lambda_cons=0.05,
    lambda_reg=1e-4,
    mc_K=4,
    w_err=1.0,
    w_unc=0.3,
    w_shift=0.3,
    w_recency=0.1,
    warmup_epochs=3,
    warmup_lr=1e-3,
    eval_batch_size=64,
    fast_dev_run=0,
)


def _build_args(conf_path, seed, gpuid, dataset, fast_dev_run,
                method=None, logname=None, backbone_method=None,
                backbone_ckpt_logname=None, backbone_seed=None, infer_chunk=None):
    lower = dataset.lower()
    p = A._build_parser()
    args = p.parse_args([])  # all parser args have defaults

    args.conf = conf_path
    args.seed = seed
    args.gpuid = gpuid
    args.dataset = dataset
    args.logname = logname or f"a2tta_lite_{lower}"
    # Backbone ckpt logname (default OL-AN; override for STAEformer etc).
    args.backbone_ckpt_logname = backbone_ckpt_logname or f"oneline_st_an_{lower}"
    args.backbone_ckpt_logname_fallback = backbone_ckpt_logname or f"retrain_st_{lower}"
    for k, v in A2TTA_HP.items():
        setattr(args, k, v)
    # Overrides. Default method = tta_ctx_local (full Ours; calibrator_arch=film
    # and local_steps=3 come from the parser/HP defaults).
    if method:
        args.method = method
    if backbone_method:
        args.backbone_method = backbone_method
    if backbone_seed is not None:
        args.backbone_seed = backbone_seed
    if infer_chunk is not None:
        args.infer_chunk = infer_chunk
    args.fast_dev_run = int(fast_dev_run)

    args.device = (
        torch.device(f"cuda:{gpuid}")
        if torch.cuda.is_available() and gpuid != -1
        else torch.device("cpu")
    )
    args.methods = A.METHOD_REGISTRY

    init(args)            # merge conf JSON into args, set args.path = log/<DS>/<logname>-<seed>
    seed_anything(args.seed)
    init_log(args)        # attach logger writing into <path>/<logname>.log
    A._stash_a2tta_cfg(args)
    return args


def evaluate(conf_path, seed, gpuid, dataset, fast_dev_run=0,
             method=None, logname=None, backbone_method=None,
             backbone_ckpt_logname=None, backbone_seed=None, infer_chunk=None):
    args = _build_args(conf_path, seed, gpuid, dataset, fast_dev_run,
                       method=method, logname=logname,
                       backbone_method=backbone_method,
                       backbone_ckpt_logname=backbone_ckpt_logname,
                       backbone_seed=backbone_seed, infer_chunk=infer_chunk)
    seed_dir = args.path
    out_csv = osp.join(seed_dir, "eval_new_sensors.csv")

    rows = []
    new_metrics, all_metrics = {}, {}
    prev = {"size": None}  # previous year's node count, tracked across the year walk

    def capturing_cal_metric(truth, pred, _args):
        """Replaces utils.metric.cal_metric inside online_a2tta_eval.

        truth/pred: [windows, N, T]. Records all-sensor metrics (sanity) and,
        when the graph grew vs the previous evaluated year, new-sensor metrics.
        """
        year = _args.year
        n = truth.shape[1]
        prev_size = prev["size"]

        am = _slice_metrics(truth, pred, slice(0, n))
        if am is not None:
            all_metrics[year] = am
            for k in am:  # every horizon key: cumulative 1..12, step1..12, Avg
                rows.append([year, n, 0, "all", k, *am[k]])

        if prev_size is not None and n > prev_size:
            new_count = n - prev_size
            nm = _slice_metrics(truth, pred, slice(prev_size, n))
            if nm is not None:
                new_metrics[year] = nm
                for k in nm:  # every horizon key: cumulative 1..12, step1..12, Avg
                    rows.append([year, n, new_count, "new", k, *nm[k]])
                _args.logger.info(
                    f"  [new-sensors] year={year} N={n} new={new_count} "
                    f"(idx {prev_size}..{n - 1})"
                )
        prev["size"] = n

    # Patch the single per-year metric call the online evaluator makes.
    a2t.cal_metric = capturing_cal_metric

    args.logger.info(
        f"[*] A2TTA new-sensor eval conf={conf_path} seed={seed} ds={dataset} "
        f"gpu={gpuid} years={args.begin_year}-{args.end_year}"
    )

    # Build the persistent (growing) calibrator and walk the years, mirroring
    # a2tta_main.main() — but without the results-CSV / pretty-print bookkeeping.
    cfg = args.a2tta
    first_graph = nx.from_numpy_array(
        np.load(osp.join(args.graph_path, f"{args.begin_year}_adj.npz"))["x"]
    )
    # Mirror a2tta_main.main()'s calibrator construction EXACTLY (incl. arch),
    # else the method/arch (e.g. tta_ctx_local + film) would silently fall back
    # to the class default and not match the main-table A2TTA.
    calibrator = ResidualCalibrator(
        num_nodes_max=first_graph.number_of_nodes(),
        x_len=args.x_len, y_len=args.y_len,
        node_emb_dim=cfg.get("node_emb_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("calibrator_dropout", 0.1),
        horizon_emb_dim=cfg.get("horizon_emb_dim", 0),
        arch=cfg.get("calibrator_arch", "film"),
        rank=cfg.get("adapter_rank", 8),
        gate_type=cfg.get("gate_type", "channel"),
        temporal_kernel=cfg.get("temporal_kernel", 3),
    ).to(args.device)

    args.graph_size_list = []
    for year in range(args.begin_year, args.end_year + 1):
        try:
            A._run_year(args, year, calibrator)
        except FileNotFoundError as e:
            args.logger.warning(f"[skip] year {year}: {e}")
            continue
        args.graph_size_list.append(args.graph_size)
        if cfg.get("fast_dev_run", False):
            args.logger.info("[fast_dev_run] stopping after first year")
            break

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "n_total", "n_new", "scope", "horizon", "MAE", "RMSE", "MAPE"])
        w.writerows(rows)

    args.logger.info("\n========== Newly added sensors — aggregate ==========")
    args.logger.info(f"years evaluated (new): {sorted(new_metrics.keys())}")
    if new_metrics:
        for k in ("3", "6", "12", "Avg"):
            vs = list(new_metrics.values())
            args.logger.info(
                "{:>3}   MAE={:>7.4f}  RMSE={:>7.4f}  MAPE={:>7.4f}".format(
                    k,
                    float(np.mean([v[k][0] for v in vs])),
                    float(np.mean([v[k][1] for v in vs])),
                    float(np.mean([v[k][2] for v in vs])),
                )
            )
    args.logger.info(f"[*] wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", required=True, help="e.g. PEMS03")
    ap.add_argument("--gpuid", type=int, default=0)
    ap.add_argument("--fast_dev_run", type=int, default=0)
    ap.add_argument("--method", default=None,
                    help="override A2TTA method (e.g. tta_ctx_local = full Ours). "
                         "Default None keeps the legacy a2tta_lite.")
    ap.add_argument("--logname", default=None,
                    help="override output logname dir (default a2tta_lite_<ds>).")
    ap.add_argument("--backbone_method", default=None,
                    help="e.g. STAEFORMER (default TrafficStream/OL-AN).")
    ap.add_argument("--backbone_ckpt_logname", default=None,
                    help="e.g. retrain_staeformer_<ds> (default oneline_st_an_<ds>).")
    ap.add_argument("--backbone_seed", type=int, default=None,
                    help="backbone ckpt seed suffix (STAE: tta_seed-9).")
    ap.add_argument("--infer_chunk", type=int, default=None,
                    help="no-grad backbone inference chunk (STAE dense graphs).")
    a = ap.parse_args()
    evaluate(a.conf, a.seed, a.gpuid, a.dataset, a.fast_dev_run,
             method=a.method, logname=a.logname,
             backbone_method=a.backbone_method,
             backbone_ckpt_logname=a.backbone_ckpt_logname,
             backbone_seed=a.backbone_seed, infer_chunk=a.infer_chunk)
