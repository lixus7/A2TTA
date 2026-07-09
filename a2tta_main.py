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
  python a2tta_main.py --conf conf/PEMS05/a2tta_olan_pems05.json \
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
    PECPM_Model, RAP_Model, STTTC_Model, STAEFORMER_Model,
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
    'STAEFORMER': STAEFORMER_Model,
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
    # Backbone ckpts may live under a different seed than the TTA seed (e.g.
    # STAEFormer retrain used seeds 42-46 while A2TTA runs 51-55). Default -1
    # keeps the legacy behaviour (same seed as the TTA run).
    backbone_seed = int(getattr(args, "backbone_seed", -1))
    if backbone_seed < 0:
        backbone_seed = args.seed

    # Set args.method for the backbone factory; restore after.
    saved_method = getattr(args, "method", None)
    saved_year = getattr(args, "year", None)
    vars(args)["method"] = backbone_method
    vars(args)["year"] = year

    def _try(logname):
        ckpt_dir = osp.join(args.model_path, f"{logname}-{backbone_seed}", str(year))
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
            f"{logname_primary}-{backbone_seed}/ or {logname_fallback}-{backbone_seed}/"
        )

    args.logger.info(f"  [backbone] year={year} loading {ckpt}")
    state = torch.load(ckpt, map_location=args.device)["model_state_dict"]
    model = METHOD_REGISTRY[backbone_method](args).to(args.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_matched = len(state) - len(unexpected)
    if n_matched == 0:
        # strict=False would otherwise leave a randomly-initialized model and
        # produce silent garbage (e.g. a TrafficStream ckpt fed to STAEFORMER).
        raise RuntimeError(
            f"Backbone ckpt {ckpt} shares NO keys with {backbone_method} "
            f"(missing={len(missing)}, unexpected={len(unexpected)}) — wrong "
            f"--backbone_ckpt_logname / --backbone_method pairing?"
        )
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
        horizon_emb_dim=cfg.get("horizon_emb_dim", 0),
        arch=cfg.get("calibrator_arch", "film"),
        rank=cfg.get("adapter_rank", 8),
        gate_type=cfg.get("gate_type", "channel"),
        temporal_kernel=cfg.get("temporal_kernel", 3),
    ).to(args.device)

    # Trainable-parameter report (ablation fairness): total, arch-head (excl.
    # node embedding), and the full list of trainable parameter names.
    _tot, _head, _named = calibrator.param_report()
    args.logger.info(
        f"[calibrator] arch={cfg.get('calibrator_arch','film')} "
        f"rank={cfg.get('adapter_rank',8)} trainable_total={_tot} "
        f"head_params(excl node_emb)={_head}")
    args.logger.info(f"[calibrator] trainable params: {_named}")

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
    p.add_argument("--conf", type=str, default="conf/PEMS05/a2tta_olan_pems05.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpuid", type=int, default=1)
    p.add_argument("--logname", type=str, default="a2tta_lite_pems05")
    p.add_argument("--method", type=str, default="tta_ctx_local",
                   choices=["backbone", "calibrator",
                            "tta_random", "tta_recent", "tta_error",
                            "tta_all", "tta_ctx_local",
                            "a2tta_lite", "a2tta_emb"],
                   help=("Which variant to run. OURS DEFAULT = tta_ctx_local. "
                         "backbone=frozen no-cal baseline; calibrator=warmed-up cal, no "
                         "online TTA; tta_random/recent/error=online-TTA selection "
                         "baselines; tta_all=adapt on all delayed labels (no selection); "
                         "tta_ctx_local=Ours, global tta_all calibrator + discardable "
                         "per-batch local context clone; a2tta_lite/a2tta_emb=legacy. "
                         "(Other explored selection ideas are archived in a2tta_back.py.)"))
    p.add_argument("--dataset", type=str, default="PEMS05")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional explicit per-year checkpoint root, ignored when "
                        "--backbone_ckpt_logname is sufficient.")
    p.add_argument("--backbone_ckpt_logname", type=str, default="oneline_st_an_pems05",
                   help="Logname under args.model_path/<logname>-<seed>/<year>/*.pkl")
    p.add_argument("--backbone_ckpt_logname_fallback", type=str, default="retrain_st_pems05")
    p.add_argument("--backbone_method", type=str, default="TrafficStream",
                   choices=list(METHOD_REGISTRY.keys()))
    p.add_argument("--backbone_seed", type=int, default=-1,
                   help="Seed suffix of the backbone ckpt dir (e.g. 42 for "
                        "retrain_staeformer_*-42 while the TTA run uses 51). "
                        "-1 = use --seed (legacy behaviour).")
    p.add_argument("--freeze_backbone", type=int, default=1)

    # Calibrator hyperparams
    p.add_argument("--adapter_hidden_dim", type=int, default=64)
    p.add_argument("--node_emb_dim", type=int, default=16)
    p.add_argument("--calibrator_dropout", type=float, default=0.1)
    p.add_argument("--calibrator_arch", type=str, default="film",
                   choices=["residual", "norm", "film", "affine", "adapter",
                            "lowrank", "gated_adapter", "temporal_conv"],
                   help=("Calibrator architecture (structural ablation). OURS = film. "
                         "residual=raw-input MLP residual (legacy); norm=EMA-standardised "
                         "MLP residual; film=conditional per-sample affine gamma*y+beta "
                         "(gamma/beta from an MLP, init identity); affine=STATIC "
                         "channel-wise (1+gamma)*y+beta (no conditioning net); "
                         "adapter=bottleneck residual y+Wup(GELU(Wdown)); lowrank=linear "
                         "low-rank residual y+U(V); gated_adapter=adapter with tanh gate; "
                         "temporal_conv=depthwise 1-D conv over the prediction horizon. "
                         "All are identity at init."))
    p.add_argument("--adapter_rank", type=int, default=8,
                   help="Bottleneck/low-rank dim for adapter/lowrank/gated_adapter.")
    p.add_argument("--gate_type", type=str, default="channel", choices=["scalar", "channel"],
                   help="gated_adapter gate granularity (default channel-wise over horizon).")
    p.add_argument("--temporal_kernel", type=int, default=3,
                   help="Kernel size for temporal_conv (depthwise over horizon).")
    # Ours selection (tta_ctx_local): per-batch local context clone.
    p.add_argument("--local_steps", type=int, default=3,
                   help="tta_ctx_local: gradient steps for the per-batch local clone.")
    p.add_argument("--horizon_emb_dim", type=int, default=0,
                   help="If >0, add a per-horizon embedding to the calibrator "
                        "(STAEFormer-inspired). 0 = legacy calibrator.")

    # a2tta_emb knobs (only used when --method a2tta_emb)
    p.add_argument("--emb_lr", type=float, default=-1.0,
                   help="LR for adaptive-embedding rows; -1 = use --adapt_lr.")
    p.add_argument("--node_budget_frac", type=float, default=0.25,
                   help="Fraction of nodes whose embedding rows may update.")
    p.add_argument("--emb_batch_cap", type=int, default=16,
                   help="Max pool samples per fresh backbone forward (memory cap).")
    p.add_argument("--lambda_reg_emb", type=float, default=-1.0,
                   help="Proximal reg for embedding rows; -1 = use --lambda_reg.")

    # Online TTA hyperparams
    p.add_argument("--adapt_lr", type=float, default=1e-3)    # Ours tuned default
    p.add_argument("--adapt_steps", type=int, default=3)      # Ours tuned default
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
    p.add_argument("--infer_chunk", type=int, default=16,
                   help="Internal sub-batch for STAEFORMER backbone inference "
                        "(memory only; logical batch / TTA cadence unchanged). "
                        "Lower for dense graphs: PEMS07→2, PEMS04/12→8.")
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
        "calibrator_arch": args.calibrator_arch,
        "adapter_rank": args.adapter_rank,
        "gate_type": args.gate_type,
        "temporal_kernel": args.temporal_kernel,
        "local_steps": args.local_steps,
        "adapt_lr": args.adapt_lr,
        "adapt_steps": args.adapt_steps,
        "adapt_every_batches": args.adapt_every_batches,
        "horizon_emb_dim": args.horizon_emb_dim,
        "emb_lr": args.emb_lr if args.emb_lr > 0 else args.adapt_lr,
        "node_budget_frac": args.node_budget_frac,
        "emb_batch_cap": args.emb_batch_cap,
        "lambda_reg_emb": args.lambda_reg_emb if args.lambda_reg_emb >= 0 else args.lambda_reg,
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
        "infer_chunk": args.infer_chunk,
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
