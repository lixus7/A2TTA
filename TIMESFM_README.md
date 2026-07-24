# TimesFM zero-shot baseline

Adds Google's **TimesFM** time-series foundation model
(<https://github.com/google-research/timesfm>) as a **zero-shot** forecasting
baseline on the xxltraffic / PEMS continual-forecasting benchmark.

> **Status: installed & smoke-tested (2026-06-04).** Model = **TimesFM 2.5**
> (`google/timesfm-2.5-200m-pytorch`).
> - conda env `timesfm` (Python 3.11) — separate from `stg`/`tsfm`.
> - timesfm 2.5 API installed editable from source at
>   `./timesfm_src` (PyPI only ships the
>   classic API, which can't load the 2.5 checkpoint).
> - checkpoint cached at `HF_HOME=./hf_cache`.
> So you can skip step 1 below and go straight to the run.

It is fully self-contained — it does **not** modify `main.py`, `stkec_main.py`,
`a2tta_main.py`, `se_lewm_main.py`, `src/model/model.py`, or any existing config.

## New files

| File | Purpose |
|------|---------|
| `timesfm_main.py` | Standalone entry point. Zero-shot eval over all years of a dataset. |
| `src/model/timesfm_wrapper.py` | Version-robust loader/forecaster (supports the classic 2.0 API **and** the newer 2.5 API). |
| `scripts/gen_timesfm_configs.py` | Generates `conf/PEMS*/timesfm_*.json` from each dataset's `retrain_st_*.json` (same years/paths). |
| `conf/PEMS*/timesfm_*.json` | Per-dataset configs (already generated). |
| `scripts/setup_timesfm_env.sh` | **One-time, login node:** create the `timesfm` conda env + pre-download the checkpoint. |
| `scripts/hpc_timesfm.pbs` | PBS job for an HPC cluster (multi-GPU, resumable). |
| `scripts/timesfm_run.sh` | Quick interactive single-dataset runner. |

## How it stays comparable to the other baselines

* **Same test set.** Every other baseline loads the cached `FastData/{year}.npz`
  test split (built with the 31-day slice → 1762 windows). TimesFM reconstructs
  the **raw-scale** context windows from `RawData/{year}.npz` using the identical
  `generate_dataset` indexing, and **verifies** the reconstruction matches the
  cached `test_y` before evaluating. (Verified bit-for-bit: `z_score(raw_x)` ==
  cached `test_x`, `raw_y` == cached `test_y`.)
* **Raw scale in, raw scale out.** TimesFM normalizes each series internally, so
  it is fed raw context (not the z-scored `test_x`) and produces raw predictions.
* **Same metrics.** Uses the shared `utils.metric.cal_metric` → MAE / RMSE / MAPE
  at horizons 3 / 6 / 12 and Avg, with `null_val=0` masking.

## Quickstart

### 1. One-time setup (on a **login node** — needs internet)

```bash
cd .
bash scripts/setup_timesfm_env.sh
```

This creates the `timesfm` conda env, installs `timesfm[torch]`, detects which
TimesFM API/checkpoint was installed, pre-downloads it into
`HF_HOME=./hf_cache`, and rewrites the
configs' `repo_id` to match.

> To pin a specific package version for reproducibility:
> `TIMESFM_PIP='timesfm[torch]==1.2.7' bash scripts/setup_timesfm_env.sh`

### 2. Smoke test (login node or interactive GPU)

```bash
conda activate timesfm
HF_HOME=./hf_cache \
  python timesfm_main.py --conf conf/PEMS03/timesfm_pems03.json \
    --gpuid 0 --seed 51 --fast_dev_run 2 --max_samples 256 \
    --csv_path run_logs/timesfm_smoke.csv
```

### 3. Full run on a PBS cluster

```bash
qsub scripts/hpc_timesfm.pbs
# subset / override:
qsub -v DATASETS="PEMS03 PEMS04",GPUS="0 1" scripts/hpc_timesfm.pbs
```

Results are written per-job to `run_logs/timesfm_<timestamp>/csv/*.csv` and
combined into `run_logs/timesfm_<timestamp>/results.csv`. Per-year metrics are
also logged to `log/<DATASET>/timesfm_<ds>-<seed>/timesfm_<ds>.log` (same layout
as the other baselines). The job runs **offline** (`HF_HUB_OFFLINE=1`) using the
pre-downloaded checkpoint, and is resumable (`-v RESUME_FROM=run_logs/timesfm_...`).

## Notes / knobs

* **Seeds:** TimesFM zero-shot is **deterministic** (no training, no sampling),
  so extra seeds reproduce identical numbers — the scripts default to one seed.
* **`context_len=32`:** our input is only 12 steps; TimesFM mask-pads it, so a
  small multiple of 32 is equivalent to a long context but far faster. Tune via
  `--context_len` or `TIMESFM_CONTEXT_LEN` when regenerating configs.
* **Checkpoint:** installed = `google/timesfm-2.5-200m-pytorch` (TimesFM 2.5,
  via the source install). To use the classic 2.0 model instead, reinstall with
  `TIMESFM_PIP='timesfm[torch]' FORCE_REPO='google/timesfm-2.0-500m-pytorch'
  bash scripts/setup_timesfm_env.sh`. Override per-run with `--repo_id`.
* **H200:** torch installed is `2.12.0+cu130` (CUDA 13). The cluster's
  `cuda/13.0.0` module confirms the H200 nodes' driver supports it; the model
  auto-runs on GPU (`cuda:0`, pinned via `--gpuid`). If a node ever reports a
  driver too old for CUDA 13, reinstall torch from a cu12x index.
* **Throughput knobs:** `--per_core_batch_size` (GPU batch) and `--chunk_size`
  (series per forecast call).
