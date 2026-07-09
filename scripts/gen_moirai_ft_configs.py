"""Generate conf/<DATASET>/moirai_ft_<kind>_<dataset>.json for the Moirai
per-year fine-tune baselines (kind in {moirai2, moe}). uni2ts has NO LoRA, so
fine-tuning is FULL (optionally backbone-frozen via freeze_encoder). Copies
years/paths from each dataset's retrain_st_<ds>.json.
Run from eac/ root:  python scripts/gen_moirai_ft_configs.py
"""
import json, os, glob

DATASETS = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07", "PEMS08",
            "PEMS10", "PEMS11", "PEMS12"]

VARIANTS = {
    # Moirai-2.0 small (~? M params): cheap -> full fine-tune, quantile (pinball) loss.
    "moirai2": dict(
        kind="moirai2",
        repo_id="Salesforce/moirai-2.0-R-small",
        context_len=12, patch_size=16, num_samples=0,
        freeze_encoder=0,        # full fine-tune
        lr=5e-5, epochs=5, batch_size=256, eval_batch_size=1024,
        num_train_windows=4096, bf16=0,
    ),
    # Moirai-MoE base (0.9B): heavy -> freeze the transformer backbone by default,
    # fine-tune in/out projections + scaler only (fast, fits 2x H200), NLL loss.
    "moe": dict(
        kind="moe",
        repo_id="Salesforce/moirai-moe-1.0-R-base",
        context_len=12, patch_size=16, num_samples=100,
        freeze_encoder=1,        # backbone frozen (0.9B); tune proj/param heads
        lr=5e-5, epochs=5, batch_size=128, eval_batch_size=512,
        num_train_windows=4096, bf16=0,
    ),
}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    n = 0
    for ds in DATASETS:
        src = sorted(glob.glob(f"conf/{ds}/retrain_st_*.json")) or sorted(glob.glob(f"conf/{ds}/eac.json"))
        if not src:
            print(f"[skip] {ds}"); continue
        b = json.load(open(src[0]))
        lower = ds.lower()
        for name, v in VARIANTS.items():
            conf = dict(
                begin_year=b["begin_year"], end_year=b["end_year"],
                x_len=b.get("x_len", 12), y_len=b.get("y_len", 12), days=31,
                raw_data_path=b["raw_data_path"], save_data_path=b["save_data_path"],
                graph_path=b["graph_path"], model_path=f"log/{ds}/",
                method=f"Moirai-{name}-FT", logname=f"moirai_ft_{name}_{lower}", **v,
            )
            out = f"conf/{ds}/moirai_ft_{name}_{lower}.json"
            json.dump(conf, open(out, "w"), indent=4)
            print(f"[ok] {out}  years={conf['begin_year']}-{conf['end_year']}")
            n += 1
    print(f"wrote {n} configs")


if __name__ == "__main__":
    main()
