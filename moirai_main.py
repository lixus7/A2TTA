"""
Moirai (uni2ts) zero-shot baseline for the xxltraffic / PEMS continual-forecasting
benchmark. Standalone entry point — does NOT touch main.py / *_main.py / src/model/model.py.

Evaluates Salesforce's Moirai foundation models in pure zero-shot mode (no training)
on EXACTLY the same per-year test windows used by every other baseline, reporting
MAE / RMSE / MAPE at horizons 3/6/12 and Avg via the shared utils.metric.cal_metric.

Two model families (selected by the conf's `kind`, mirrors timesfm_main.py):
  kind="moirai2" -> Moirai-2.0-R-small   (quantile decoder; point = 0.5 quantile)
  kind="moe"     -> Moirai-MoE base       (sample model;  point = sample mean)

Correctness (identical to timesfm_main.py):
  Moirai standardises each series internally, so it needs RAW-scale context. We
  reconstruct the raw context windows from RawData/{year}.npz with the same 31-day
  slice + split + generate_dataset() indexing the cached FastData was built with,
  and VERIFY the reconstructed raw targets match the cached test_y before trusting
  the alignment. Ground-truth targets come straight from the cached test_y.

Run:
    python moirai_main.py --conf conf/PEMS03/moirai_moirai2_pems03.json --gpuid 0
"""

import sys, os, argparse, time, csv, json, random, logging
sys.path.append("src/")

import numpy as np
import os.path as osp


def _mkdirs(p):
    os.makedirs(p, exist_ok=True)


def _merge_conf(args):
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
    except Exception:
        pass


def _init_log(args):
    logger = logging.getLogger("moirai_main")
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
    """(S,L,N) raw context, (S,H,N) ground-truth targets (== cached test_y).
    Reconstructed with the same 31-day slice + split + windowing as FastData,
    verified against the cached raw test_y on a row subsample."""
    from utils.data_convert import generate_dataset

    raw = np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"]
    fast_path = osp.join(args.save_data_path, f"{year}.npz")
    have_fast = osp.exists(fast_path)
    ref_x = test_y = None
    if have_fast:
        fast = np.load(fast_path, allow_pickle=True)
        ref_x, test_y = fast["test_x"], fast["test_y"]

    days = int(getattr(args, "days", 31))
    MAX_CELLS = 600_000_000

    def reconstruct(slice_days):
        data = raw[0:days * 288, :] if slice_days else raw
        t, n = data.shape[0], data.shape[1]
        test_idx = [i for i in range(int(t * 0.8), t)]
        if len(test_idx) < args.x_len + args.y_len + 1:
            return None, None
        if len(test_idx) * n * args.x_len > MAX_CELLS:
            return None, None
        rx, ry = generate_dataset(data, test_idx, args.x_len, args.y_len)
        ry = np.nan_to_num(ry, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return rx.astype(np.float32), ry

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
        f"shape={None if test_y is None else test_y.shape}; tried {tried}.")


def main(args):
    from utils.metric import cal_metric
    from model.moirai_wrapper import MoiraiForecaster

    args.result = {k: {" MAE": {}, "MAPE": {}, "RMSE": {}} for k in ("3", "6", "12", "Avg")}

    t0 = time.time()
    forecaster = MoiraiForecaster(
        kind=args.kind, repo_id=args.repo_id, context_len=args.context_len,
        horizon_len=args.y_len, patch_size=int(getattr(args, "patch_size", 16)),
        num_samples=int(getattr(args, "num_samples", 100)),
        device=("cuda" if _cuda() else "cpu"),
        dtype=getattr(args, "dtype", "float32"), logger=args.logger)
    args.logger.info(f"[Moirai-{args.kind}] model ready in {time.time()-t0:.1f}s "
                     f"(repo={args.repo_id}, patch={forecaster.patch_size})")

    years = list(range(args.begin_year, args.end_year + 1))
    if args.fast_dev_run:
        years = years[:args.fast_dev_run]
        args.logger.info(f"[fast_dev_run] years -> {years}")

    for year in years:
        vars(args)["year"] = year
        raw_ctx, truth, info = build_context_and_truth(args, year)  # (S,L,N),(S,H,N)
        S, L, N = raw_ctx.shape
        H = truth.shape[1]
        if args.max_samples and S > args.max_samples:
            raw_ctx, truth = raw_ctx[:args.max_samples], truth[:args.max_samples]
            S = raw_ctx.shape[0]

        ctx = np.transpose(raw_ctx, (0, 2, 1)).reshape(S * N, L)     # (S*N, L)

        tstart = time.time()
        preds = forecaster.forecast(ctx, horizon=H, chunk_size=args.chunk_size)  # (S*N, H)
        elapsed = time.time() - tstart

        preds = preds.reshape(S, N, H)
        gt = np.transpose(truth, (0, 2, 1))                          # (S, N, H)
        args.logger.info(
            f"[*] year {year}: {info}, series={S*N} ({S} x {N}), "
            f"forecast {elapsed:.1f}s ({S*N/max(elapsed,1e-9):.0f} series/s)")
        cal_metric(gt, preds, args)

    args.logger.info("\n\n[Moirai-%s zero-shot summary] %s", args.kind, args.logname)
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            vals, line = [], ""
            for year in years:
                if year in args.result[i][j]:
                    line += "{:>10.2f}\t".format(args.result[i][j][year])
                    vals.append(args.result[i][j][year])
            args.logger.info("{:<4}\t{}\t{}\t{:>8.2f}".format(i, j, line, np.mean(vals) if vals else float("nan")))

    if args.csv_path:
        write_csv(args, years)


def write_csv(args, years):
    _mkdirs(osp.dirname(osp.abspath(args.csv_path)))
    header = ["dataset", "method", "kind", "seed", "repo_id", "year"]
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
            row = [getattr(args, "dataset", "?"), args.method, args.kind, args.seed,
                   args.repo_id, year]
            for i in ["3", "6", "12", "Avg"]:
                for j in [" MAE", "RMSE", "MAPE"]:
                    row.append(round(float(args.result[i][j].get(year, float("nan"))), 4))
            w.writerow(row)
    args.logger.info(f"[Moirai-{args.kind}] wrote results -> {args.csv_path}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, required=True)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--logname", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--csv_path", type=str, default=None)
    # Moirai knobs (override conf)
    p.add_argument("--kind", type=str, default=None, choices=[None, "moirai2", "moe"])
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--context_len", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None, choices=[None, "float32", "bfloat16", "float16"])
    p.add_argument("--chunk_size", type=int, default=1024)
    # eval scope (dev)
    p.add_argument("--fast_dev_run", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    if args.gpuid is not None and args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    cli_keys = ["logname", "dataset", "csv_path", "kind", "repo_id",
                "context_len", "patch_size", "num_samples", "dtype"]
    cli = {k: getattr(args, k) for k in cli_keys if getattr(args, k) is not None}
    _merge_conf(args)
    for k, v in cli.items():
        setattr(args, k, v)

    defaults = dict(kind="moirai2", repo_id="Salesforce/moirai-2.0-R-small",
                    context_len=12, patch_size=16, num_samples=100, dtype="float32",
                    method="Moirai", dataset="?", csv_path=None)
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    args.path = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    _mkdirs(args.path)
    _seed_everything(args.seed)
    _init_log(args)
    args.logger.info("params : %s", {k: v for k, v in vars(args).items() if k != "logger"})
    main(args)
