"""
TimesFM zero-shot baseline for the xxltraffic / PEMS continual-forecasting benchmark.

Standalone entry point — does NOT touch main.py / stkec_main.py / a2tta_main.py etc.
It evaluates Google's TimesFM time-series foundation model in pure zero-shot mode
(no training, no fine-tuning) on exactly the same per-year test windows used by all
other baselines, and reports MAE / RMSE / MAPE at horizons 3/6/12 and Avg via the
shared utils.metric.cal_metric.

Key correctness points
-----------------------
1. Same test set as every other baseline. The other baselines load the cached
   FastData/{year}.npz test split. TimesFM, being a foundation model, normalizes
   each input series internally, so it needs *raw-scale* context (not the z-scored
   test_x). We therefore reconstruct the raw context windows from RawData/{year}.npz
   using the identical generate_dataset() indexing, and we VERIFY that z_score(raw_x)
   reproduces the cached test_x bit-for-bit before trusting the alignment. The
   ground-truth targets are taken straight from the cached FastData test_y, so the
   evaluation is identical to the other baselines.

2. Raw context in, raw predictions out. cal_metric() masks out null_val=0 targets,
   matching the rest of the benchmark.

Run:
    python timesfm_main.py --conf conf/PEMS03/timesfm_pems03.json --gpuid 0 --seed 51
"""

import sys, os, argparse, time, csv, json, random, logging
sys.path.append("src/")

import numpy as np
import os.path as osp


# --- tiny standalone scaffolding ----------------------------------------------
# Deliberately NOT importing utils.initialize / utils.common_tools: that module
# has a top-level `from Bio.Cluster import kcluster` (biopython, used by the
# clustering baselines) which is irrelevant to TimesFM and would force biopython
# into this env. We only reuse the numpy-only shared pieces that guarantee an
# identical evaluation: utils.data_convert.generate_dataset and utils.metric.cal_metric.

def _mkdirs(p):
    os.makedirs(p, exist_ok=True)


def _merge_conf(args):
    """Merge the JSON conf into args (mirrors utils.initialize.init: every key
    except gpuid)."""
    with open(args.conf) as f:
        info = json.load(f)
    for k, v in info.items():
        if k != "gpuid":
            setattr(args, k, v)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _init_log(args):
    logger = logging.getLogger("timesfm_main")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    _mkdirs(args.path)
    fh = logging.FileHandler(osp.join(args.path, args.logname + ".log"))
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    for h in (fh, ch):
        h.setFormatter(fmt)
        logger.addHandler(h)
    args.logger = logger
    logger.info("log file: %s", osp.join(args.path, args.logname + ".log"))


def build_context_and_truth(args, year):
    """
    Returns
      raw_ctx : (S, x_len, N) float32, raw-scale context windows for TimesFM.
      truth   : (S, y_len, N) float32, ground-truth targets (== cached test_y).
      info    : str describing how alignment was achieved.

    The cached FastData test split (used by all baselines) was generated with the
    31-day slice active (31*288 = 8928 timesteps). We reconstruct raw windows with
    the same slice + split, and validate alignment by checking the reconstructed
    raw targets against the cached (raw-scale) test_y on a row subsample.
    """
    from utils.data_convert import generate_dataset

    raw = np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"]
    fast_path = osp.join(args.save_data_path, f"{year}.npz")
    have_fast = osp.exists(fast_path)
    ref_x = test_y = None
    if have_fast:
        fast = np.load(fast_path, allow_pickle=True)
        ref_x, test_y = fast["test_x"], fast["test_y"]

    days = int(getattr(args, "days", 31))
    # Safety cap: never reconstruct an absurdly large window tensor (the no-slice
    # fallback over a full ~105k-step year x thousands of nodes would OOM). The
    # real (sliced) split is ~1762 windows, so this only blocks the bad branch.
    MAX_CELLS = 600_000_000  # ~2.4 GB float32

    def reconstruct(slice_days):
        data = raw[0:days * 288, :] if slice_days else raw
        t, n = data.shape[0], data.shape[1]
        test_idx = [i for i in range(int(t * 0.8), t)]  # train_rate(0.6)+val_rate(0.2)
        if len(test_idx) < args.x_len + args.y_len + 1:
            return None, None
        if len(test_idx) * n * args.x_len > MAX_CELLS:
            return None, None  # refuse to build a memory bomb
        rx, ry = generate_dataset(data, test_idx, args.x_len, args.y_len)
        ry = np.nan_to_num(ry, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return rx.astype(np.float32), ry

    # The cached FastData split was generated with the 31-day slice, so try that
    # first and return as soon as it aligns. Alignment is verified directly and
    # cheaply against the cached raw-scale test_y on a row subsample (no global
    # z-score over tens of millions of elements).
    tried = []
    for slice_days in (True, False):
        rx, ry = reconstruct(slice_days)
        if rx is None:
            continue
        if not have_fast:
            return rx, ry, f"computed(no-fast, slice={slice_days})"
        tried.append((slice_days, rx.shape))
        if rx.shape == ref_x.shape == (ry.shape[0],) + ref_x.shape[1:]:
            k = min(256, ry.shape[0])
            if np.allclose(ry[:k], test_y[:k], atol=1e-3):
                return rx, test_y.astype(np.float32), f"aligned(slice={slice_days})"

    raise RuntimeError(
        f"[year {year}] could not align reconstructed raw context with cached test_y "
        f"shape={None if test_y is None else test_y.shape}; tried {tried}. "
        f"Refusing to evaluate on a mismatched test set."
    )


def impute_context_nan(ctx):
    """Per-series NaN/inf imputation with the series mean (0 if all-missing).
    TimesFM normalizes by the context mean, so mean-imputation is scale-neutral."""
    ctx = np.array(ctx, dtype=np.float32, copy=True)
    bad = ~np.isfinite(ctx)
    if bad.any():
        row_mean = np.nanmean(np.where(np.isfinite(ctx), ctx, np.nan), axis=1)
        row_mean = np.nan_to_num(row_mean, nan=0.0)
        ctx[bad] = np.take(row_mean, np.where(bad)[0])
    return ctx


def main(args):
    from utils.metric import cal_metric
    from model.timesfm_wrapper import TimesFMForecaster

    args.result = {k: {" MAE": {}, "MAPE": {}, "RMSE": {}} for k in ("3", "6", "12", "Avg")}

    t0 = time.time()
    forecaster = TimesFMForecaster(
        repo_id=args.repo_id,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        per_core_batch_size=args.per_core_batch_size,
        backend=args.backend,
        logger=args.logger,
    )
    args.logger.info(f"[TimesFM] model ready in {time.time() - t0:.1f}s (api={forecaster.api})")

    years = list(range(args.begin_year, args.end_year + 1))
    if args.fast_dev_run:
        years = years[:args.fast_dev_run]
        args.logger.info(f"[fast_dev_run] limiting to years {years}")

    for year in years:
        vars(args)["year"] = year
        raw_ctx, truth, info = build_context_and_truth(args, year)  # (S,L,N), (S,H,N)
        S, L, N = raw_ctx.shape
        H = truth.shape[1]

        if args.max_samples and S > args.max_samples:
            raw_ctx, truth = raw_ctx[:args.max_samples], truth[:args.max_samples]
            S = raw_ctx.shape[0]

        # (S, L, N) -> (S*N, L) univariate context windows
        ctx = np.transpose(raw_ctx, (0, 2, 1)).reshape(S * N, L)
        ctx = impute_context_nan(ctx)

        tstart = time.time()
        preds = forecaster.forecast(ctx, horizon=H, chunk_size=args.chunk_size)  # (S*N, H)
        elapsed = time.time() - tstart

        preds = preds.reshape(S, N, H)
        gt = np.transpose(truth, (0, 2, 1))  # (S, N, H)  -- layout cal_metric expects

        args.logger.info(
            f"[*] year {year}: {info}, series={S * N} ({S} samples x {N} nodes), "
            f"forecast {elapsed:.1f}s ({S * N / max(elapsed, 1e-9):.0f} series/s)"
        )
        cal_metric(gt, preds, args)

    # ---- summary table (mirrors main.py) ----
    args.logger.info("\n\n[TimesFM zero-shot summary] %s", args.logname)
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            vals, info = [], ""
            for year in years:
                if year in args.result[i][j]:
                    info += "{:>10.2f}\t".format(args.result[i][j][year])
                    vals.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t{}\t{:>8.2f}".format(i, j, info, np.mean(vals) if vals else float("nan")))

    if args.csv_path:
        write_csv(args, years)


def write_csv(args, years):
    os.makedirs(osp.dirname(osp.abspath(args.csv_path)), exist_ok=True)
    header = ["dataset", "method", "seed", "repo_id", "year"]
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            header.append(f"T{i}_{j.strip()}")
    new = not osp.exists(args.csv_path)
    with open(args.csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        for year in years:
            if year not in args.result["Avg"][" MAE"]:
                continue
            row = [getattr(args, "dataset", "?"), "TimesFM", args.seed, args.repo_id, year]
            for i in ["3", "6", "12", "Avg"]:
                for j in [" MAE", "RMSE", "MAPE"]:
                    row.append(round(float(args.result[i][j].get(year, float("nan"))), 4))
            w.writerow(row)
    args.logger.info(f"[TimesFM] wrote results -> {args.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default=None, help="override logname from conf")
    parser.add_argument("--dataset", type=str, default=None, help="label only (for CSV)")
    parser.add_argument("--csv_path", type=str, default=None)
    # TimesFM knobs (override conf)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--horizon_len", type=int, default=None)
    parser.add_argument("--per_core_batch_size", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=[None, "gpu", "cpu", "torch"])
    parser.add_argument("--chunk_size", type=int, default=4096)
    # eval scope
    parser.add_argument("--fast_dev_run", type=int, default=0, help="N>0: only first N years")
    parser.add_argument("--max_samples", type=int, default=0, help="N>0: cap test samples/year (dev)")
    args = parser.parse_args()

    # Pin the GPU before importing torch/timesfm so the model lands on cuda:0.
    if args.gpuid is not None and args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    # Capture knobs explicitly set on the CLI (argparse default = None) BEFORE
    # merging the conf json, because conf is allowed to overwrite args.
    _cli_keys = ["logname", "dataset", "csv_path", "repo_id", "context_len",
                 "horizon_len", "per_core_batch_size", "backend"]
    cli_overrides = {k: getattr(args, k) for k in _cli_keys if getattr(args, k) is not None}

    _merge_conf(args)  # loads conf json into args (sets repo_id/context_len/... + logname)

    # CLI wins over conf for the knobs the user passed explicitly.
    for k, v in cli_overrides.items():
        setattr(args, k, v)

    # Hard defaults for anything still unset (neither conf nor CLI provided it).
    defaults = dict(repo_id="google/timesfm-2.5-200m-pytorch", context_len=32,
                    horizon_len=128, per_core_batch_size=512, backend="gpu",
                    dataset="?", csv_path=None)
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    args.path = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    _mkdirs(args.path)
    _seed_everything(args.seed)
    _init_log(args)
    args.logger.info("params : %s", {k: v for k, v in vars(args).items() if k != "logger"})
    main(args)
