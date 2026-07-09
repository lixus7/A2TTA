"""Generate conf/<DATASET>/timesfm_ft_<dataset>.json for the TimesFM 2.5 LoRA
fine-tune baseline. Copies years/paths from each dataset's retrain_st_<ds>.json.
Run from eac/ root:  python scripts/gen_timesfm_ft_configs.py
"""
import json, os, glob

DATASETS = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07", "PEMS08",
            "PEMS10", "PEMS11", "PEMS12"]

FT = dict(
    repo_id="google/timesfm-2.5-200m-transformers",  # transformers ckpt (PEFT path)
    context_len=32,          # patch length; 32 real history steps (see main docstring)
    lora_r=8, lora_alpha=16, lora_dropout=0.05,
    lr=1e-4, epochs=5, batch_size=256, eval_batch_size=512,
    num_train_windows=4096,  # random train windows sampled per year
    bf16=1,
)


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
        conf = dict(
            begin_year=b["begin_year"], end_year=b["end_year"],
            x_len=b.get("x_len", 12), y_len=b.get("y_len", 12), days=31,
            raw_data_path=b["raw_data_path"], save_data_path=b["save_data_path"],
            graph_path=b["graph_path"], model_path=f"log/{ds}/",
            method="TimesFM-FT", logname=f"timesfm_ft_{lower}", **FT,
        )
        out = f"conf/{ds}/timesfm_ft_{lower}.json"
        json.dump(conf, open(out, "w"), indent=4)
        print(f"[ok] {out}  years={conf['begin_year']}-{conf['end_year']}")
        n += 1
    print(f"wrote {n} configs")


if __name__ == "__main__":
    main()
