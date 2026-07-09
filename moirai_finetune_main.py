"""
Moirai (uni2ts) per-year fine-tune baseline (retrain-style) for the xxltraffic /
PEMS continual-forecasting benchmark. Standalone — does NOT touch the main files.

Unlike the zero-shot baseline (moirai_main.py), this FINE-TUNES the model on each
year's few-shot (20%) train split and evaluates on that year's test split,
mirroring the "retrain" baselines. uni2ts has NO LoRA, so this is FULL fine-tune
(optionally with the transformer backbone frozen for the 0.9B MoE). Because there
is genuine training (random data order, dropout, optimiser noise), DIFFERENT SEEDS
GIVE DIFFERENT RESULTS -> we run 3 seeds and report mean +/- std.

Two model families (conf `kind`):
  kind="moirai2" -> Moirai-2.0-R-small  : pinball (quantile) loss on the future patch.
  kind="moe"     -> Moirai-MoE base     : PackedNLLLoss on the predictive distribution
                                          (the uni2ts MoiraiFinetune objective).

Context length = 12 (same lookback as the benchmark / zero-shot column); Moirai
left-pads to its patch size internally, so the evaluated test windows and targets
are identical to the other baselines (no lookback caveat).

Run (per dataset+seed; loops all years internally):
    python moirai_finetune_main.py --conf conf/PEMS03/moirai_ft_moirai2_pems03.json \
        --gpuid 0 --seed 51 --csv_path run_logs/moirai_ft.csv
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
    logger = logging.getLogger("moirai_ft_main")
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


def build_year_data(args, year):
    """train_ctx (n_tr,L) train_tgt (n_tr,H) | test_ctx (S*N,L) test_truth (S,N,H) | N,S.
    Built from the same 31-day slice + 20%/test split as the benchmark."""
    from utils.data_convert import generate_dataset

    raw = np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"]
    days = int(getattr(args, "days", 31))
    data = raw[0:days * 288, :]
    t, N = data.shape[0], data.shape[1]
    L, H = int(args.context_len), int(args.y_len)

    # TRAIN windows from the few-shot 20% train portion
    tr_end = int(t * float(getattr(args, "train_frac", 0.2)))   # 0.2=few-shot; 0.6=match baselines
    tr = data[0:tr_end, :]
    txr, tyr = generate_dataset(tr, list(range(tr.shape[0])), L, H)
    Xtr = np.transpose(txr, (0, 2, 1)).reshape(-1, L)
    Ytr = np.transpose(tyr, (0, 2, 1)).reshape(-1, H)
    good = np.isfinite(Xtr).all(1) & np.isfinite(Ytr).all(1)
    Xtr, Ytr = Xtr[good], Ytr[good]
    n_keep = min(int(args.num_train_windows), Xtr.shape[0])
    if Xtr.shape[0] > n_keep:
        idx = np.random.choice(Xtr.shape[0], n_keep, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]

    # TEST windows: identical prediction points as the benchmark
    test_idx = list(range(int(t * 0.8), t))
    xte, yte = generate_dataset(data, test_idx, L, H)
    S = xte.shape[0]
    yte = np.nan_to_num(yte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_ctx = np.transpose(xte, (0, 2, 1)).reshape(-1, L).astype(np.float32)
    test_truth = np.transpose(yte, (0, 2, 1)).astype(np.float32)            # (S,N,H)
    return Xtr.astype(np.float32), Ytr.astype(np.float32), test_ctx, test_truth, N, S


def finetune_one_year(forecaster, params, Xtr, Ytr, args):
    import torch
    n = Xtr.shape[0]
    bs = int(args.batch_size)
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=0.01)
    steps = max(1, int(args.epochs) * math_ceil(n, bs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    forecaster.module.train()
    last = 0.0
    rng = np.random.default_rng(args.seed)
    for ep in range(int(args.epochs)):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            sel = order[s:s + bs]
            loss = forecaster.loss(Xtr[sel], Ytr[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            last = float(loss.item())
    return last


def math_ceil(a, b):
    return (a + b - 1) // b


def main(args):
    import torch
    from utils.metric import cal_metric
    from model.moirai_wrapper import MoiraiForecaster

    args.result = {k: {" MAE": {}, "MAPE": {}, "RMSE": {}} for k in ("3", "6", "12", "Avg")}

    t0 = time.time()
    forecaster = MoiraiForecaster(
        kind=args.kind, repo_id=args.repo_id, context_len=args.context_len,
        horizon_len=args.y_len, patch_size=int(getattr(args, "patch_size", 16)),
        num_samples=int(getattr(args, "num_samples", 100)),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype="float32", logger=args.logger)
    params = forecaster.trainable_parameters(freeze_encoder=bool(int(getattr(args, "freeze_encoder", 0))))
    trn = sum(p.numel() for p in params)
    tot = sum(p.numel() for p in forecaster.module.parameters())
    args.logger.info(f"[Moirai-{args.kind}-FT] ready in {time.time()-t0:.1f}s; "
                     f"trainable {trn/1e6:.2f}M / {tot/1e6:.1f}M "
                     f"(freeze_encoder={getattr(args,'freeze_encoder',0)})")

    # snapshot pretrained weights to reset per year (retrain-style)
    init_state = {k: v.detach().cpu().clone() for k, v in forecaster.module.state_dict().items()}

    years = list(range(args.begin_year, args.end_year + 1))
    if args.fast_dev_run:
        years = years[:args.fast_dev_run]
        args.logger.info(f"[fast_dev_run] years -> {years}")

    # year-level resume: skip years already written to the csv (lets a 100h job be
    # resubmitted with the same csv_path and pick up where it timed out).
    done_years = _csv_done_years(args)
    if done_years:
        skip = [y for y in years if y in done_years]
        years = [y for y in years if y not in done_years]
        args.logger.info(f"[resume] skipping {len(skip)} done years {skip}; remaining {years}")

    for year in years:
        vars(args)["year"] = year
        Xtr, Ytr, test_ctx, test_truth, N, S = build_year_data(args, year)
        if args.max_test_samples and S > args.max_test_samples:
            S = args.max_test_samples
            test_truth = test_truth[:S]
            test_ctx = test_ctx[:S * N]

        forecaster.module.load_state_dict(init_state)   # fresh start each year
        ts = time.time()
        last = finetune_one_year(forecaster, params, Xtr, Ytr, args)
        t_train = time.time() - ts

        ts = time.time()
        preds = forecaster.forecast(test_ctx, horizon=int(args.y_len),
                                    chunk_size=int(args.eval_batch_size))  # (S*N,H)
        t_eval = time.time() - ts
        preds = preds.reshape(S, N, int(args.y_len))
        args.logger.info(
            f"[*] year {year}: train {Xtr.shape[0]} win (loss {last:.4f}, {t_train:.1f}s) | "
            f"eval {S}x{N} ({t_eval:.1f}s)")
        cal_metric(test_truth, preds, args)
        if args.csv_path:
            _append_year_row(args, year)   # persist immediately (year-level resume)

    args.logger.info("\n\n[Moirai-%s fine-tune summary] %s seed=%s", args.kind, args.logname, args.seed)
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            vals = [args.result[i][j][y] for y in years if y in args.result[i][j]]
            line = "".join("{:>9.2f}".format(args.result[i][j][y]) for y in years if y in args.result[i][j])
            args.logger.info("{:<4}\t{}\t{}\t{:>8.2f}".format(i, j, line, np.mean(vals) if vals else float("nan")))


def _csv_header():
    header = ["dataset", "method", "kind", "seed", "repo_id", "year"]
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            header.append(f"T{i}_{j.strip()}")
    return header


def _csv_done_years(args):
    """Years already present in csv_path for THIS seed (for year-level resume)."""
    if not args.csv_path or not osp.exists(args.csv_path):
        return set()
    done = set()
    try:
        with open(args.csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if str(r.get("seed")) == str(args.seed) and r.get("kind") == args.kind:
                    try:
                        done.add(int(r["year"]))
                    except (ValueError, TypeError, KeyError):
                        pass
    except Exception:
        return set()
    return done


def _append_year_row(args, year):
    """Append one year's metrics, flock-guarded (multiple jobs share run dirs)."""
    import fcntl
    if year not in args.result["Avg"][" MAE"]:
        return
    _mkdirs(osp.dirname(osp.abspath(args.csv_path)))
    new = not osp.exists(args.csv_path)
    row = [args.dataset, args.method, args.kind, args.seed, args.repo_id, year]
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            row.append(round(float(args.result[i][j].get(year, float("nan"))), 4))
    with open(args.csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.writer(f)
        if new:
            w.writerow(_csv_header())
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, required=True)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--logname", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--csv_path", type=str, default=None)
    # model / training knobs (override conf)
    p.add_argument("--kind", type=str, default=None, choices=[None, "moirai2", "moe"])
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--context_len", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--freeze_encoder", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--num_train_windows", type=int, default=None)
    p.add_argument("--train_frac", type=float, default=0.2,
                   help="fraction of per-year window for FT training (0.2=few-shot; 0.6=match baselines)")
    # eval scope (dev)
    p.add_argument("--fast_dev_run", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    args = p.parse_args()

    if args.gpuid is not None and args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    cli_keys = ["logname", "dataset", "csv_path", "kind", "repo_id", "context_len",
                "patch_size", "num_samples", "freeze_encoder", "lr", "epochs",
                "batch_size", "eval_batch_size", "num_train_windows"]
    cli = {k: getattr(args, k) for k in cli_keys if getattr(args, k) is not None}
    _merge_conf(args)
    for k, v in cli.items():
        setattr(args, k, v)

    defaults = dict(kind="moirai2", repo_id="Salesforce/moirai-2.0-R-small",
                    context_len=12, patch_size=16, num_samples=100, freeze_encoder=0,
                    lr=5e-5, epochs=5, batch_size=256, eval_batch_size=1024,
                    num_train_windows=4096, method="Moirai-FT", dataset="?", csv_path=None)
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    args.path = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    _mkdirs(args.path)
    _seed_everything(args.seed)
    _init_log(args)
    args.logger.info("params : %s", {k: v for k, v in vars(args).items() if k != "logger"})
    main(args)
