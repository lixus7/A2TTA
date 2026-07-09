"""
TimesFM 2.5 LoRA fine-tune baseline (per-year, retrain-style) for the
xxltraffic / PEMS continual-forecasting benchmark.

Standalone — does NOT touch main.py / *_main.py / src/model/model.py.

Unlike the zero-shot baseline (timesfm_main.py), this FINE-TUNES a LoRA adapter
on each year's few-shot (20%) train split and evaluates on that year's test
split — mirroring the "retrain" baselines. Because there is genuine training
(LoRA random init, data shuffling, dropout, AdamW), DIFFERENT SEEDS GIVE
DIFFERENT RESULTS, so we run 3 seeds and report mean +/- std.

Implementation notes
--------------------
* Uses the HuggingFace Transformers TimesFM 2.5 model
  (`google/timesfm-2.5-200m-transformers`) + PEFT LoRA, following the official
  example timesfm_src/.../examples/finetuning/finetune_lora.py:
      out = model(past_values=ctx, future_values=tgt, forecast_context_len=L)
      loss = out.loss                       # MSE + quantile, normalized space
      preds = out.mean_predictions[:, :H]   # point forecast
* CONTEXT LENGTH = 32 (the model's patch length). The transformers model
  requires the context to be a multiple of 32, and front zero-padding corrupts
  its internal RevIN mean/std (computed over all positions). So we feed 32 REAL
  history steps. The benchmark/zero-shot use a 12-step lookback; the fine-tune
  column therefore uses a 32-step lookback (noted in the paper table caption).
  The forecast horizon and the evaluated test targets are unchanged (12 steps),
  so MAE/RMSE/MAPE are computed on the same prediction points and remain
  comparable. (A handful of windows at the very start of the test region are
  dropped because they lack 32 prior steps; immaterial to the aggregate.)
* TimesFM normalizes each series internally (RevIN), so we feed RAW-scale
  context and targets, exactly like the zero-shot baseline.

Run (per dataset+seed; loops all years internally):
    python timesfm_finetune_main.py --conf conf/PEMS03/timesfm_ft_pems03.json \
        --gpuid 0 --seed 51 --csv_path run_logs/timesfm_ft.csv
"""

import sys, os, argparse, time, csv, json, random, logging
sys.path.append("src/")

import numpy as np
import os.path as osp


# --- tiny standalone scaffolding (see timesfm_main.py for rationale) ----------
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
    logger = logging.getLogger("timesfm_ft_main")
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


# --- data ---------------------------------------------------------------------
def _impute_rows(x):
    """Per-row (per-series) NaN/inf -> row mean (0 if all-missing)."""
    x = np.array(x, dtype=np.float32, copy=True)
    bad = ~np.isfinite(x)
    if bad.any():
        rm = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=1)
        rm = np.nan_to_num(rm, nan=0.0)
        x[bad] = np.take(rm, np.where(bad)[0])
    return x


def build_year_data(args, year):
    """
    Returns
      train_ctx (n_tr, ctx_len)  train_tgt (n_tr, horizon)   -- sampled raw windows
      test_ctx  (S*N, ctx_len)   test_truth (S, N, horizon)  -- benchmark test targets
      N (int)
    Built from the same 31-day slice + 20%/test split as the benchmark, but with a
    ctx_len-step context (32) instead of 12.
    """
    from utils.data_convert import generate_dataset

    raw = np.load(osp.join(args.raw_data_path, f"{year}.npz"))["x"]
    days = int(getattr(args, "days", 31))
    data = raw[0:days * 288, :]
    t = data.shape[0]
    N = data.shape[1]
    # `cl` = real lookback used to BUILD windows (e.g. 12, matching baselines/ZS).
    # The model is still fed forecast_context_len=context_len (a 32-multiple);
    # short contexts are padded to 32 and the mask-aware RevIN patch keeps the
    # normalization stats over the real steps only. lookback<=0 -> context_len.
    _lb = int(getattr(args, "lookback", 0))
    cl = _lb if _lb > 0 else int(args.context_len)
    H = int(args.y_len)

    # ---- TRAIN: windows from the train portion (train_frac of the window) ----
    tr_end = int(t * float(getattr(args, "train_frac", 0.2)))   # 0.2 few-shot; 0.6 match baselines
    tr_data = data[0:tr_end, :]
    txr, tyr = generate_dataset(tr_data, list(range(tr_data.shape[0])), cl, H)  # (n,cl,N),(n,H,N)
    # flatten to per-(window,node)
    Xtr = np.transpose(txr, (0, 2, 1)).reshape(-1, cl)
    Ytr = np.transpose(tyr, (0, 2, 1)).reshape(-1, H)
    # drop windows with any non-finite value (clean training targets)
    good = np.isfinite(Xtr).all(1) & np.isfinite(Ytr).all(1)
    Xtr, Ytr = Xtr[good], Ytr[good]
    # subsample
    n_keep = min(int(args.num_train_windows), Xtr.shape[0])
    if Xtr.shape[0] > n_keep:
        idx = np.random.choice(Xtr.shape[0], n_keep, replace=False)
        Xtr, Ytr = Xtr[idx], Ytr[idx]

    # ---- TEST: same prediction points as the benchmark, 32-step context -----
    test_idx = list(range(int(t * 0.8), t))
    xte, yte = generate_dataset(data, test_idx, cl, H)  # (S,cl,N),(S,H,N)
    S = xte.shape[0]
    yte = np.nan_to_num(yte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    test_ctx = _impute_rows(np.transpose(xte, (0, 2, 1)).reshape(-1, cl))  # (S*N, cl)
    test_truth = np.transpose(yte, (0, 2, 1)).astype(np.float32)            # (S, N, H)
    return (Xtr.astype(np.float32), Ytr.astype(np.float32),
            test_ctx.astype(np.float32), test_truth, N, S)


# --- model (transformers + PEFT LoRA) -----------------------------------------
def load_base_and_lora(args):
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    from peft import LoraConfig, get_peft_model

    # mask-aware RevIN so a short (e.g. 12-step) context padded to a 32-multiple
    # keeps correct per-window normalization stats. No-op for full 32-real ctx.
    import timesfm_revin_patch
    timesfm_revin_patch.apply(getattr(args, "logger", None))

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = TimesFm2_5ModelForPrediction.from_pretrained(args.repo_id, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    lcfg = LoraConfig(r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
                      target_modules="all-linear", lora_dropout=float(args.lora_dropout),
                      bias="none")
    model = get_peft_model(model, lcfg)
    # snapshot the fresh LoRA init so each year can reset to it
    lora_init = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}
    return model, lora_init, device


def reset_lora(model, lora_init):
    model.load_state_dict(lora_init, strict=False)


def finetune_one_year(model, device, Xtr, Ytr, args):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(args.lr), weight_decay=0.01)
    steps = max(1, int(args.epochs) * len(dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    last = 0.0
    for ep in range(int(args.epochs)):
        for xb, yb in dl:
            xb = xb.to(device, dtype=dtype); yb = yb.to(device, dtype=dtype)
            out = model(past_values=xb, future_values=yb, forecast_context_len=int(args.context_len))
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); sched.step()
            last = float(loss.item())
    return last


def forecast_test(model, device, test_ctx, H, args):
    import torch
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model.eval()
    bs = int(args.eval_batch_size)
    out = []
    with torch.no_grad():
        for s in range(0, test_ctx.shape[0], bs):
            xb = torch.from_numpy(test_ctx[s:s + bs]).to(device, dtype=dtype)
            pred = model(past_values=xb, forecast_context_len=int(args.context_len)).mean_predictions
            out.append(pred[:, :H].float().cpu().numpy())
    return np.concatenate(out, 0)  # (S*N, H)


# --- main ---------------------------------------------------------------------
def main(args):
    from utils.metric import cal_metric

    args.result = {k: {" MAE": {}, "MAPE": {}, "RMSE": {}} for k in ("3", "6", "12", "Avg")}

    t0 = time.time()
    model, lora_init, device = load_base_and_lora(args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    args.logger.info(f"[TimesFM-FT] model ready in {time.time()-t0:.1f}s on {device}; "
                     f"LoRA trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M  (r={args.lora_r})")

    years = list(range(args.begin_year, args.end_year + 1))
    if args.fast_dev_run:
        years = years[:args.fast_dev_run]
        args.logger.info(f"[fast_dev_run] years -> {years}")

    for year in years:
        vars(args)["year"] = year
        Xtr, Ytr, test_ctx, test_truth, N, S = build_year_data(args, year)
        if args.max_test_samples and S > args.max_test_samples:
            S = args.max_test_samples
            test_truth = test_truth[:S]
            test_ctx = test_ctx[:S * N]

        reset_lora(model, lora_init)            # fresh adapter for this year (retrain-style)
        ts = time.time()
        last_loss = finetune_one_year(model, device, Xtr, Ytr, args)
        t_train = time.time() - ts

        ts = time.time()
        preds = forecast_test(model, device, test_ctx, int(args.y_len), args)  # (S*N, H)
        t_eval = time.time() - ts

        preds = preds.reshape(S, N, int(args.y_len))
        args.logger.info(
            f"[*] year {year}: train {Xtr.shape[0]} win (loss {last_loss:.4f}, {t_train:.1f}s) | "
            f"eval {S}x{N} ({t_eval:.1f}s)")
        cal_metric(test_truth, preds, args)

    # ---- summary ----
    args.logger.info("\n\n[TimesFM-2.5 LoRA fine-tune summary] %s seed=%s", args.logname, args.seed)
    for i in ["3", "6", "12", "Avg"]:
        for j in [" MAE", "RMSE", "MAPE"]:
            vals = [args.result[i][j][y] for y in years if y in args.result[i][j]]
            line = "".join("{:>9.2f}".format(args.result[i][j][y]) for y in years if y in args.result[i][j])
            args.logger.info("{:<4}\t{}\t{}\t{:>8.2f}".format(i, j, line, np.mean(vals) if vals else float("nan")))

    if args.csv_path:
        write_csv(args, years)


def write_csv(args, years):
    _mkdirs(osp.dirname(osp.abspath(args.csv_path)))
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
            row = [args.dataset, "TimesFM-FT", args.seed, args.repo_id, year]
            for i in ["3", "6", "12", "Avg"]:
                for j in [" MAE", "RMSE", "MAPE"]:
                    row.append(round(float(args.result[i][j].get(year, float("nan"))), 4))
            w.writerow(row)
    args.logger.info(f"[TimesFM-FT] wrote results -> {args.csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--conf", type=str, required=True)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--gpuid", type=int, default=0)
    p.add_argument("--logname", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--csv_path", type=str, default=None)
    # model / LoRA / training knobs (override conf)
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--context_len", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--num_train_windows", type=int, default=None)
    p.add_argument("--bf16", type=int, default=None)
    p.add_argument("--lookback", type=int, default=0,
                   help="real history steps to build windows (e.g. 12, matching baselines/ZS); "
                        "0=use context_len (legacy 32-step). Model still fed forecast_context_len=context_len.")
    p.add_argument("--train_frac", type=float, default=0.2,
                   help="fraction of per-year window for FT training (0.2 few-shot; 0.6 match baselines)")
    # eval scope (dev)
    p.add_argument("--fast_dev_run", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=0)
    args = p.parse_args()

    if args.gpuid is not None and args.gpuid >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

    cli_keys = ["logname", "dataset", "csv_path", "repo_id", "context_len", "lora_r",
                "lora_alpha", "lora_dropout", "lr", "epochs", "batch_size",
                "eval_batch_size", "num_train_windows", "bf16"]
    cli = {k: getattr(args, k) for k in cli_keys if getattr(args, k) is not None}
    _merge_conf(args)
    for k, v in cli.items():
        setattr(args, k, v)

    defaults = dict(repo_id="google/timesfm-2.5-200m-transformers", context_len=32,
                    lora_r=8, lora_alpha=16, lora_dropout=0.05, lr=1e-4, epochs=5,
                    batch_size=256, eval_batch_size=512, num_train_windows=4096,
                    bf16=1, dataset="?", csv_path=None)
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    args.path = osp.join(args.model_path, args.logname + "-" + str(args.seed))
    _mkdirs(args.path)
    _seed_everything(args.seed)
    _init_log(args)
    args.logger.info("params : %s", {k: v for k, v in vars(args).items() if k != "logger"})
    main(args)
