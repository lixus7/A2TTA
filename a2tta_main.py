"""
A2TTA-Lite: Active Adaptive Test-Time Adaptation for Traffic Forecasting.

Entry point that orchestrates per-year evaluation:
  * loads a frozen backbone checkpoint (default: Online-AN per-year .pkls)
  * builds a residual calibrator (zero-init) shared across years
  * optionally warms up the calibrator on the year's train+val split
  * runs causal online TTA on the year's test split with delayed-label
    active selection
  * logs metrics via the same `cal_metric` used by main.py and writes a CSV

Usage:
  python a2tta_main.py --conf conf/PEMS05/a2tta_lite_pems05.json \
      --method a2tta_lite --seed 42 --gpuid 1 --backbone_ckpt_logname oneline_st_an_pems05

See `scripts/a2tta_lite_pems05_run.sh` for the full experiment matrix.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

sys.path.append("src/")

import numpy as np
import networkx as nx
import torch

from utils.data_convert import generate_samples
from utils.initialize import init, seed_anything, init_log
from utils.common_tools import mkdirs

from src.model.model import (
    TrafficStream_Model, STKEC_Model, EAC_Model, Universal_Model,
    STGNN_Model, DCRNN_Model, ASTGNN_Model, TGCN_Model,
    PECPM_Model, RAP_Model, STTTC_Model,
)
from src.model.a2tta import ResidualCalibrator
from src.trainer.a2tta_trainer import (
    warmup_calibrator, online_a2tta_eval, append_results_csv,
)
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from torch_geometric.loader import DataLoader


METHOD_REGISTRY = {
    'TrafficStream': TrafficStream_Model,
    'STKEC': STKEC_Model,
    'EAC': EAC_Model,
    'Universal': Universal_Model,
    'STGNN': STGNN_Model,
    'DCRNN': DCRNN_Model,
    'ASTGNN': ASTGNN_Model,
    'TGCN': TGCN_Model,
    'PECPM': PECPM_Model,
    'RAP': RAP_Model,
    'STTTC': STTTC_Model,
}


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def _load_backbone_for_year(args, year: int):
    """Load the per-year backbone checkpoint produced by `oneline_st_an_pems05`
    (or the configured logname). Falls back to retrain ckpt if Online-AN is
    missing. Returns a torch.nn.Module on `args.device`.
    """
    backbone_method = getattr(args, "backbone_method", "TrafficStream")
    logname_primary = args.backbone_ckpt_logname
    logname_fallback = getattr(args, "backbone_ckpt_logname_fallback", "retrain_st_pems05")

    # Set args.method for the backbone factory; restore after.
    saved_method = getattr(args, "method", None)
    saved_year = getattr(args, "year", None)
    vars(args)["method"] = backbone_method
    vars(args)["year"] = year

    def _try(logname):
        ckpt_dir = osp.join(args.model_path, f"{logname}-{args.seed}", str(year))
        if not osp.isdir(ckpt_dir):
            return None
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pkl")]
        if not files:
            return None
        # Min loss in filename (matches load_best_model logic).
        files = sorted(files, key=lambda f: float(f[:-4]))
        return osp.join(ckpt_dir, files[0])

    ckpt = _try(logname_primary) or _try(logname_fallback)
    if ckpt is None:
        raise FileNotFoundError(
            f"No backbone .pkl for year={year} under "
            f"{logname_primary}-{args.seed}/ or {logname_fallback}-{args.seed}/"
        )

    args.logger.info(f"  [backbone] year={year} loading {ckpt}")
    state = torch.load(ckpt, map_location=args.device)["model_state_dict"]
    model = METHOD_REGISTRY[backbone_method](args).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Restore.
    vars(args)["method"] = saved_method
    vars(args)["year"] = saved_year
    return model


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

def _run_year(args, year: int, calibrator: ResidualCalibrator):
    # --- Graph + adjacency ---
    graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"])
    if year == args.begin_year:
        vars(args)["base_node_size"] = graph.number_of_nodes()
        vars(args)["init_graph_size"] = graph.number_of_nodes()
    vars(args)["graph_size"] = graph.number_of_nodes()
    vars(args)["year"] = year

    adj = np.load(osp.join(args.graph_path, f"{year}_adj.npz"))["x"]
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

    # --- Inputs (preprocessed FastData / RawData) ---
    inputs = (
        generate_samples(31, osp.join(args.save_data_path, str(year)),
                         np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"],
                         graph, val_test_mix=False)
        if args.data_process
        else np.load(osp.join(args.save_data_path, f"{year}.npz"), allow_pickle=True)
    )
    args.logger.info(f"[*] Year {year} loaded")

    # --- Backbone (frozen) ---
    backbone = _load_backbone_for_year(args, year)

    # --- Calibrator (grow node-emb if graph grew) ---
    calibrator.expand_nodes(graph.number_of_nodes())
    calibrator = calibrator.to(args.device)

    cfg = args.a2tta

    # --- Optional warm-up ---
    if cfg.get("warmup_epochs", 0) > 0 and cfg["method"] != "backbone":
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"),
                                  batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"),
                                batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, num_workers=0)
        warmup_calibrator(
            backbone=backbone, calibrator=calibrator,
            train_loader=train_loader, val_loader=val_loader,
            args=args, epochs=cfg["warmup_epochs"], lr=cfg.get("warmup_lr", 1e-3),
        )

    # --- Online evaluation ---
    online_a2tta_eval(backbone, calibrator, inputs, args)

    # Free per-year backbone.
    del backbone
    torch.cuda.empty_cache()


def main(args):
    args.logger.info("params : %s", vars(args))
    args.result = {
        "3":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "6":   {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "12":  {" MAE": {}, "MAPE": {}, "RMSE": {}},
        "Avg": {" MAE": {}, "MAPE": {}, "RMSE": {}},
    }
    mkdirs(args.save_data_path)

    # Build a calibrator once and let it persist (and grow) across years.
    cfg = args.a2tta
    # Initialize with the year-0 graph size; will expand if later years grow.
    first_graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, f"{args.begin_year}_adj.npz"))["x"])
    calibrator = ResidualCalibrator(
        num_nodes_max=first_graph.number_of_nodes(),
        x_len=args.x_len, y_len=args.y_len,
        node_emb_dim=cfg.get("node_emb_dim", 16),
        hidden_dim=cfg.get("hidden_dim", 64),
        dropout=cfg.get("calibrator_dropout", 0.1),
    ).to(args.device)

    # Track graph sizes per year (mirrors main.py's args.graph_size_list).
    vars(args)["graph_size_list"] = []

    for year in range(args.begin_year, args.end_year + 1):
        try:
            _run_year(args, year, calibrator)
        except FileNotFoundError as e:
            args.logger.warning(f"[skip] year {year}: {e}")
            continue
        args.graph_size_list.append(args.graph_size)
        if cfg.get("fast_dev_run", False):
            args.logger.info("[fast_dev_run] stopping after first year")
            break

    # Pretty-print final table (mirrors main.py).
    args.logger.info("\n\n")
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            info, vals = "", []
            for year in range(args.begin_year, args.end_year + 1):
                if year in args.result[i][j]:
                    info += "{:>10.2f}\t".format(args.result[i][j][year])
                    vals.append(args.result[i][j][year])
            if vals:
                args.logger.info("{:<4}\t{}\t".format(i, j) + info + "\t{:>8.2f}".format(np.mean(vals)))

    # Persist per-year per-horizon rows to CSV.
    csv_path = cfg.get("csv_path", "run_logs/a2tta_lite_results.csv")
    csv_path = osp.join(osp.dirname(__file__), csv_path) if not osp.isabs(csv_path) else csv_path
    append_results_csv(args, csv_path)
    args.logger.info(f"[a2tta] CSV → {csv_path}")


def _build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, default="conf/PEMS05/a2tta_lite_pems05.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpuid", type=int, default=1)
    p.add_argument("--logname", type=str, default="a2tta_lite_pems05")
    p.add_argument("--method", type=str, default="a2tta_lite",
                   choices=["backbone", "calibrator",
                            "tta_random", "tta_recent", "tta_error",
                            "a2tta_lite", "tta_all"],
                   help=("Which variant to run. backbone=frozen no-cal baseline; "
                         "calibrator=warmed-up cal, no online TTA; tta_*=online TTA "
                         "with random / recent / error-only / full-active selection; "
                         "tta_all=delayed-label upper bound (uses full pool, no select)."))
    p.add_argument("--dataset", type=str, default="PEMS05")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional explicit per-year checkpoint root, ignored when "
                        "--backbone_ckpt_logname is sufficient.")
    p.add_argument("--backbone_ckpt_logname", type=str, default="oneline_st_an_pems05",
                   help="Logname under args.model_path/<logname>-<seed>/<year>/*.pkl")
    p.add_argument("--backbone_ckpt_logname_fallback", type=str, default="retrain_st_pems05")
    p.add_argument("--backbone_method", type=str, default="TrafficStream",
                   choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--freeze_backbone", type=int, default=1)

    # Calibrator hyperparams
    p.add_argument("--adapter_hidden_dim", type=int, default=64)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--calibrator_dropout", type=float, default=0.1)

    # Online TTA hyperparams
    p.add_argument("--adapt_lr", type=float, default=3e-4)
    p.add_argument("--adapt_steps", type=int, default=1)
    p.add_argument("--adapt_every_batches", type=int, default=1)
    p.add_argument("--budget_frac", type=float, default=0.25)
    p.add_argument("--candidate_pool_size", type=int, default=512)
    p.add_argument("--lambda_cons", type=float, default=0.05)
    p.add_argument("--lambda_reg", type=float, default=1e-4)
    p.add_argument("--mc_K", type=int, default=4)

    # Active scoring weights
    p.add_argument("--w_err", type=float, default=1.0)
    p.add_argument("--w_unc", type=float, default=0.3)
    p.add_argument("--w_shift", type=float, default=0.3)
    p.add_argument("--w_recency", type=float, default=0.1)

    # Warm-up
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_lr", type=float, default=1e-3)

    # Eval-time loader / CSV / dev knobs
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--csv_path", type=str, default="run_logs/a2tta_lite_results.csv")
    p.add_argument("--fast_dev_run", type=int, default=0)
    return p


def _stash_a2tta_cfg(args):
    """Pull all A2TTA-Lite knobs into a single `args.a2tta` dict for the trainer."""
    args.a2tta = {
        "dataset": args.dataset,
        "method": args.method,
        "hidden_dim": args.adapter_hidden_dim,
        "node_emb_dim": args.node_emb_dim,
        "calibrator_dropout": args.calibrator_dropout,
        "adapt_lr": args.adapt_lr,
        "adapt_steps": args.adapt_steps,
        "adapt_every_batches": args.adapt_every_batches,
        "budget_frac": args.budget_frac,
        "candidate_pool_size": args.candidate_pool_size,
        "lambda_cons": args.lambda_cons,
        "lambda_reg": args.lambda_reg,
        "mc_K": args.mc_K,
        "w_err": args.w_err,
        "w_unc": args.w_unc,
        "w_shift": args.w_shift,
        "w_recency": args.w_recency,
        "warmup_epochs": args.warmup_epochs,
        "warmup_lr": args.warmup_lr,
        "eval_batch_size": args.eval_batch_size,
        "csv_path": args.csv_path,
        "fast_dev_run": bool(args.fast_dev_run),
    }


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    vars(args)["device"] = (
        torch.device(f"cuda:{args.gpuid}")
        if torch.cuda.is_available() and args.gpuid != -1
        else torch.device("cpu")
    )

    # Methods registry consumed by `_load_backbone_for_year` via args.methods.
    vars(args)["methods"] = METHOD_REGISTRY

    # Pull JSON conf into args (same as main.py / utils.initialize.init).
    init(args)
    seed_anything(args.seed)
    init_log(args)
    _stash_a2tta_cfg(args)
    main(args)
