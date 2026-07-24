# Moirai (uni2ts) baselines for evo-xxltraffic

Zero-shot **and** per-year fine-tuned baselines for two Salesforce Moirai
time-series foundation models, wired into the `eac/` PEMS continual-forecasting
benchmark the same standalone way as the Chronos-2 / TimesFM baselines — **no
main files touched** (`main.py`, `*_main.py`, `src/model/model.py`, trainer).

Models:
- **Moirai-2.0-R-small** (`Salesforce/moirai-2.0-R-small`) — `kind=moirai2`.
  Quantile decoder; point forecast = the **0.5 quantile** (what GluonTS's
  `QuantileForecast.mean` falls back to).
- **Moirai-MoE base** (`Salesforce/moirai-moe-1.0-R-base`, 0.9B) — `kind=moe`.
  Sample model; point forecast = **mean over `num_samples` draws**.

## Files
| file | role |
|---|---|
| `moirai_main.py` | zero-shot evaluator (both kinds) |
| `moirai_finetune_main.py` | per-year fine-tune evaluator (both kinds), 3 seeds |
| `src/model/moirai_wrapper.py` | `MoiraiForecaster`: load + `forecast()` + `loss()` for both kinds |
| `scripts/gen_moirai_configs.py` | writes `conf/PEMS*/moirai_{moirai2,moe}_*.json` |
| `scripts/gen_moirai_ft_configs.py` | writes `conf/PEMS*/moirai_ft_{moirai2,moe}_*.json` |
| `scripts/setup_moirai_env.sh` | one-time env + checkpoint download |
| `scripts/hpc_moirai.pbs` | zero-shot PBS sweep (multi-GPU, resumable) |
| `scripts/hpc_moirai_ft.pbs` | fine-tune PBS sweep (multi-GPU, year-level resume) |

## Environment
Dedicated conda env **`moirai`** (Python 3.11). uni2ts pins `torch>=2.1,<2.5`,
incompatible with the `stg`/`chronos`/`timesfm` envs (torch 2.12), hence a
separate env. `uni2ts==2.0.0` (the first PyPI release shipping Moirai-2.0) +
`torch==2.4.1+cu121` (supports H200 / sm_90). Checkpoints cached under
`HF_HOME=./hf_cache`; offline HPC jobs run
`HF_HUB_OFFLINE=1`.

```bash
bash scripts/setup_moirai_env.sh        # ONCE on a login node (needs internet)
```

## Correctness
Same test set as every other baseline: raw context windows are reconstructed
from `RawData/{year}.npz` with the identical 31-day slice + `[0.8t, t)` split +
`generate_dataset` indexing the cached `FastData` was built with, and verified
against the cached raw `test_y` before evaluating. Metrics use the shared
`utils.metric.cal_metric` (cumulative-horizon MAE/RMSE/MAPE at T=3/6/12/Avg,
`null_val=0` mask). Moirai standardises each series internally (packed scaler),
so it is fed **raw**-scale context. **Univariate** per (window, node) series
(like the TimesFM baseline); `context_len=12` matches the benchmark lookback —
the model left-pads 12→patch_size(16) internally, so the evaluated windows/targets
are bit-identical to the other baselines.

We bypass the GluonTS `PandasDataset`/predictor plumbing (far too slow for the
millions of series here) and call `forecast.forward` directly on batched tensors
built exactly like `Moirai2Forecast.predict`.

## Fine-tuning
uni2ts has **no LoRA** and ships **no** Moirai-2.0 / Moirai-MoE finetune module,
so we full-fine-tune per (dataset, year) on that year's 20% train split
(retrain-style), with these objectives — each the model family's natural loss:
- **moirai2**: pinball (quantile) loss on the predicted future patch, in the
  model's scaled space — the exact quantity the inference path reads out.
- **moe**: `PackedNLLLoss` on the predictive distribution — the objective used
  by uni2ts's shipped `MoiraiFinetune.training_step`.

The 0.9B MoE backbone (`module.encoder`) is **frozen by default**
(`freeze_encoder=1` in its conf) — only the in/out projections, scaler, and
distribution heads train, keeping per-year fine-tune cheap on 2 GPUs. Moirai-2.0
small is fully fine-tuned. Fine-tune is seed-sensitive → **3 seeds (51 52 53)**,
report mean ± std.

## Run
```bash
# zero-shot (deterministic for moirai2; tiny sampling variance for moe -> 1 seed)
qsub scripts/hpc_moirai.pbs
qsub -v MODELS="moirai2" scripts/hpc_moirai.pbs        # one model only

# fine-tune (3 seeds; year-level resume across 100h resubmissions)
qsub scripts/hpc_moirai_ft.pbs
qsub -v RESUME_FROM=run_logs/moirai_ft_<TS> scripts/hpc_moirai_ft.pbs   # resume

# local smoke (CPU ok): 1 dataset, 2 years, capped windows
HF_HOME=./hf_cache \
  python moirai_main.py --conf conf/PEMS03/moirai_moirai2_pems03.json \
    --gpuid 0 --fast_dev_run 2 --max_samples 64 --csv_path run_logs/moirai_smoke.csv
```

Outputs: per-job CSVs under `run_logs/moirai[_ft]_<TS>/csv/`, combined into
`results.csv`; per-(dataset,year) logs under `.../logs/`. CSV columns include
`kind` so the two models coexist in one file.
