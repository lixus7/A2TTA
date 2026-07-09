"""
Generate conf/<DATASET>/timesfm_<dataset>.json for the TimesFM zero-shot baseline.

We copy the canonical data paths / year ranges from each dataset's existing
retrain_st_<ds>.json so TimesFM evaluates on exactly the same years and splits.
Run from the eac/ root:  python scripts/gen_timesfm_configs.py
"""
import json, os, glob

DATASETS = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07", "PEMS08",
            "PEMS10", "PEMS11", "PEMS12"]

# TimesFM checkpoint + inference settings (override per-run on the CLI if desired).
# repo_id can be overridden via the TIMESFM_REPO env var so setup_timesfm_env.sh
# can pin it to whatever package version it installs (classic 2.0 vs 2.5).
TIMESFM = dict(
    repo_id=os.environ.get("TIMESFM_REPO", "google/timesfm-2.0-500m-pytorch"),
    # context window the model is configured for. Our input is only 12 steps;
    # TimesFM pads it (mask-attended), so a small multiple of 32 is equivalent
    # to a long context here but ~16x faster than 512.
    context_len=int(os.environ.get("TIMESFM_CONTEXT_LEN", "32")),
    horizon_len=128,      # compiled max horizon (>= y_len=12)
    per_core_batch_size=int(os.environ.get("TIMESFM_BATCH", "512")),
    backend="gpu",
)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    written = []
    for ds in DATASETS:
        src = sorted(glob.glob(f"conf/{ds}/retrain_st_*.json")) or sorted(glob.glob(f"conf/{ds}/eac.json"))
        if not src:
            print(f"[skip] {ds}: no source conf found")
            continue
        base = json.load(open(src[0]))
        lower = ds.lower()
        conf = dict(
            begin_year=base["begin_year"],
            end_year=base["end_year"],
            x_len=base.get("x_len", 12),
            y_len=base.get("y_len", 12),
            days=31,                       # FastData test split was built with the 31-day slice
            raw_data_path=base["raw_data_path"],
            save_data_path=base["save_data_path"],
            graph_path=base["graph_path"],
            model_path=f"log/{ds}/",
            method="TimesFM",
            logname=f"timesfm_{lower}",
            **TIMESFM,
        )
        out = f"conf/{ds}/timesfm_{lower}.json"
        with open(out, "w") as f:
            json.dump(conf, f, indent=4)
        written.append(out)
        print(f"[ok] {out}  years={conf['begin_year']}-{conf['end_year']}")
    print(f"\nwrote {len(written)} configs")


if __name__ == "__main__":
    main()
