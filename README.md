<div align="center">
  <h2><b>A2TTA: Active Adaptive Test-Time Adaptation <br> for Continual Traffic Forecasting under Extreme Sensor Growth</b></h2>
</div>

<div align="center">

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

</div>

> **TL;DR.** Evolving-graph continual forecasters degrade sharply when the sensor network
> grows by orders of magnitude over its lifetime (e.g. **+9433%** on PEMS05). A2TTA keeps
> **any frozen backbone** and attaches a tiny **per-node FiLM calibrator** adapted **at test
> time** from **delayed ground-truth labels**; at each step it additionally spins up a
> **discardable local clone** specialised on context-weighted recent labels. It recovers
> accuracy at a fraction of the cost of retraining and is **backbone-agnostic** — validated on
> both the **Online-AN** and **STAEformer** backbones.

<p align="center">
  <!-- 🚧 TODO: methodology figure coming soon (see notebook/ours_method_spec.md for the spec). -->
  <em>🚧 Methodology figure coming soon — spec in <a href="./notebook/ours_method_spec.md"><code>notebook/ours_method_spec.md</code></a>.</em>
</p>

---

## 📖 Method

A2TTA decouples *what the model knows* (a frozen backbone) from *how it adapts on the fly*
(a lightweight calibrator), so adaptation cost stays constant even as the graph explodes.
**The wrapper is backbone-agnostic** — it consumes only the backbone's prediction, the input
window, and node ids, so any pretrained forecaster plugs in unchanged (we report both
**Online-AN / TrafficStream** and **STAEformer** backbones).

1. **Frozen backbone.** A per-year checkpoint (`--backbone_method`, default **Online-AN**;
   **STAEformer** also supported) is loaded and frozen — no backbone gradients at test time.
   It emits a raw-scale `H`-step forecast `y_base`.

2. **FiLM calibrator** ([`src/model/a2tta.py`](src/model/a2tta.py), `--calibrator_arch film`).
   A small per-node MLP consuming `y_base`, the input window `x_in`, four temporal statistics
   (last / mean / std / OLS-slope) and a learnable per-node embedding, emitting a per-horizon
   affine `ŷ = γ ⊙ y_base + β` with `γ = 1 + 0.5·tanh(·)` (init 1) and `β = std·β_norm`
   (init 0). The head is **zero-init**, so the calibrator is the **identity at start** — it
   can only help, never hurt, before adaptation. The node table **grows automatically** as
   new sensors appear. (Legacy archs `residual / affine / adapter / …` remain selectable.)

3. **Delayed-label online adaptation** ([`src/trainer/a2tta_trainer.py`](src/trainer/a2tta_trainer.py)).
   Test windows are processed in **true chronological order**; a window's ground truth is only
   revealed after its horizon `H` has physically elapsed — enforced by a
   `pending → candidate_pool` queue, so **no label leakage** is possible. Each batch, the
   **persistent** calibrator takes a few steps on the delayed-label pool (supervised L1 +
   optional consistency + proximal-to-init).

4. **Discardable local clone** ([`src/trainer/ctx_local.py`](src/trainer/ctx_local.py),
   `--method tta_ctx_local` — *our method*). Before predicting each batch, the stable global
   calibrator is **cloned**; the clone takes a few steps on pool samples **re-weighted by
   relevance to the current window** (time-of-day / day-of-week phase · input+base-prediction
   cosine similarity · recency, softmax-normalised with an ESS guard), predicts the batch, and
   is then **discarded** — so the global calibrator is never biased by any single context. On
   free delayed-label TTA this is the *only* mechanism we found that consistently beats
   adapting on all labels (all other sample-selection schemes are ablated below).

5. **Identical metrics.** Predictions are scored with the same `cal_metric` used by every
   baseline, so A2TTA numbers are directly comparable.

### Ablation-D (`--method` × `--calibrator_arch`)

Each row of Ablation-D is one `a2tta_main.py` config (numbers in
[`tables/ablation.md`](tables/ablation.md)):

| `--method` | `--calibrator_arch` | Description |
|---|---|---|
| `backbone` | — | frozen backbone, no calibration (lower bound) |
| `calibrator` | `film` | warmed-up FiLM calibrator, **no** online TTA |
| `tta_all` | `film` | online TTA on the full delayed-label pool, no local clone |
| `tta_ctx_local` | `film` | **full A2TTA (Ours)** — global TTA + discardable local clone |
| `tta_ctx_local` | `affine` | Ours with a static affine calibrator instead of FiLM |

The selection-study modes (`tta_random / tta_recent / tta_error`, and active `a2tta_lite`)
also remain available on the `--method` flag.

---

## 📦 Repository layout

```
a2tta/
├── main.py                 # entry for all backbones / continual baselines
├── a2tta_main.py           # entry for A2TTA (+ its ablation variants)
├── stkec_main.py           # entry for STKEC
├── src/
│   ├── model/              # model.py (all backbones+baselines), a2tta.py (FiLM calibrator), ...
│   ├── trainer/            # a2tta_trainer.py (online loop), ctx_local.py (local clone), ...
│   └── dataer/             # SpatioTemporalDataset.py
├── utils/                  # data_convert, initialize, metric, common_tools
├── conf/                   # per-dataset JSON configs (a2tta_olan_*, a2tta_stae_*, baselines)
├── scripts/                # runners for A2TTA + baselines + analysis helpers
├── tables/                 # result tables — main / ablation-D / new-sensor (md + tex + PDF)
├── notebook/               # figure notebooks (HP sensitivity, new-sensor, per-horizon,
│                           #   dataset maps) + render scripts + method spec
├── data/                   # dataset skeleton + processing notebooks (see data/README.md)
└── environment.yaml
```

---

## 🚀 Getting started

### 1. Environment

```bash
conda env create -f environment.yaml
conda activate stg
```

Core dependencies: `python`, `pytorch`, `torch-geometric`, `networkx`, `scipy`, `numpy`,
`tqdm`. A single CUDA GPU is enough; CPU works for debugging (`--gpuid -1`).

### 2. Data

The processed tensors are large and are released separately — see
**[`data/README.md`](data/README.md)**

> ☁️ **Cloud-disk download link:** `https://pan.baidu.com/s/1llz16kYY33TrWlKENNHC5A?pwd=xxtf code: xxtf`
> ☁️ **Raw data Link:**   `https://pan.baidu.com/s/1BPuxL96npWlfRXDv38duww?pwd=xxtf code: xxtf`

Place them under `data/<dataset>/{RawData,FastData,graph}/`; the configs already point there.
Datasets: **XXL expanding-sensor** benchmarks
`pems03 … pems12` (2005–2025), where the sensor count grows by up to two orders of magnitude.

---

## 🏃 Running A2TTA

> **Prerequisite.** A2TTA adapts on top of a *frozen per-year backbone*. By default it loads
> the **Online-AN** checkpoints. Make sure you have run the Online-AN stage for the dataset
> first (it is step 5 of every `scripts/pemsXX_run.sh`; for PEMS05 you can also run
> `python main.py --conf conf/PEMS05/oneline_st_an_pems05.json --gpuid 0 --seed 51`).

**Single run — full method (Ours), Online-AN backbone** (PEMS05, one seed):

```bash
python a2tta_main.py \
    --conf conf/PEMS05/a2tta_olan_pems05.json \
    --method tta_ctx_local --calibrator_arch film \
    --dataset PEMS05 --backbone_method TrafficStream --freeze_backbone 1 \
    --backbone_ckpt_logname oneline_st_an_pems05 --backbone_seed 51 \
    --gpuid 0 --seed 51
```

**Same, on the STAEformer backbone** (`backbone_seed = seed − 9` for the STAE checkpoints):

```bash
python a2tta_main.py \
    --conf conf/PEMS05/a2tta_stae_pems05.json \
    --method tta_ctx_local --calibrator_arch film \
    --dataset PEMS05 --backbone_method STAEFORMER --freeze_backbone 1 \
    --backbone_ckpt_logname retrain_staeformer_pems05 --backbone_seed 42 \
    --gpuid 0 --seed 51
```

**Ablation-D** — sweep the five rows by changing only `--method` / `--calibrator_arch`:

```bash
for cfg in "backbone -" "calibrator film" "tta_all film" \
           "tta_ctx_local film" "tta_ctx_local affine"; do
  set -- $cfg
  python a2tta_main.py --conf conf/PEMS05/a2tta_olan_pems05.json --dataset PEMS05 \
      --backbone_method TrafficStream --backbone_ckpt_logname oneline_st_an_pems05 \
      --backbone_seed 51 --method "$1" --calibrator_arch "$2" --gpuid 0 --seed 51
done
```

**Full ablation matrix** on PEMS05 (all 6 variants × 5 seeds):

```bash
bash scripts/a2tta_lite_pems05_run.sh
# single seed / GPU:        GPU=0 SEEDS="51" bash scripts/a2tta_lite_pems05_run.sh
# only the main method:     METHODS="a2tta_lite" bash scripts/a2tta_lite_pems05_run.sh
# quick sanity (1yr,4 bat): FAST_DEV_RUN=1 bash scripts/a2tta_lite_pems05_run.sh
```

**All datasets** (`a2tta_lite_all_datasets_6gpu.sh` dispatches jobs across the GPUs you give it):

```bash
GPUS="0 1" DATASETS="PEMS03 PEMS04" bash scripts/a2tta_lite_all_datasets_6gpu.sh
```

Outputs:
- per-year metric logs → `log/<DATASET>/a2tta_*-<seed>/*.log`
- aggregated results CSV → `run_logs/a2tta_lite_*_results.csv` (year × method × seed × horizon)
- summarize a CSV into a table with `python scripts/a2tta_summarize.py <csv>`

Key knobs (env-overridable in the script, or CLI flags on `a2tta_main.py`):
`ADAPT_LR`, `ADAPT_STEPS`, `BUDGET_FRAC`, `POOL_SIZE`, `LOCAL_STEPS`, `WARMUP_EPOCHS`,
`LAMBDA_CONS`, `LAMBDA_REG`, `HIDDEN_DIM`, `NODE_EMB_DIM`, and the active-score weights
`--w_err / --w_unc / --w_shift / --w_recency`. Defaults (from the sensitivity study):
`adapt_lr 1e-3 · adapt_steps 3 · candidate_pool_size 512 · budget_frac 0.25 · warmup 3 ·
local_steps 3`; only `adapt_lr` is materially sensitive (flat optimum `1e-3–3e-3`).

---

## 📈 Results & figures

Result tables (Markdown + LaTeX + a zoomable **vector PDF** rendering) live in
[`tables/`](tables/):

| Table | File |
|---|---|
| Main results (12 models × 9 datasets, A2TTA on **both** backbones) | [`tables/main_table.md`](tables/main_table.md) · [`.pdf`](tables/main_table.pdf) |
| **Ablation-D** (component knock-out, both backbones) | [`tables/ablation.md`](tables/ablation.md) · [`.pdf`](tables/ablation.pdf) |
| New-sensor generalisation (`a2tta-olan` / `a2tta-staef`) | [`tables/tsas_new_sensors.md`](tables/tsas_new_sensors.md) · [`.pdf`](tables/tsas_new_sensors.pdf) |

Figures are reproducible notebooks in [`notebook/`](notebook/) (each embeds its rendered PNG so
it displays without re-running; regeneration needs the released `run_logs/` summaries at repo
root — see [`notebook/README.md`](notebook/README.md)):

| Notebook | Figure |
|---|---|
| [`hyper-a2tta.ipynb`](notebook/hyper-a2tta.ipynb) | HP-sensitivity (OAT, 6 knobs, both backbones) |
| [`new_sensor_baselines.ipynb`](notebook/new_sensor_baselines.ipynb) | new-sensor error vs baselines |
| [`per_horizon_lines.ipynb`](notebook/per_horizon_lines.ipynb) | per-horizon error curves |
| `render_final.py` (pems-grid) + `fig_churn` | sensor geographic maps & yearly sensor churn |

The method write-up used to design the overview figure is
[`notebook/ours_method_spec.md`](notebook/ours_method_spec.md).

---

## 📊 Running the baselines

All baselines reported in the main table are launched from `scripts/`. Each runner takes the
same env overrides: `GPU=<id>`, `SEEDS="..."`, `DATASETS="..."`, `METHODS="..."`,
and `NOHUP=1` to background with a timestamped log under `run_logs/`.

| Group (main-table column) | Methods | How to run |
|---|---|---|
| **Naïve schemes** | Pretrain, Retrain, Online-NN, Online-AN | `bash scripts/pemsXX_run.sh` (steps 1–5) |
| **Evolving-graph continual** | TrafficStream, STKEC, EAC | `bash scripts/pemsXX_run.sh` (steps 6–8) |
| **Static STGNN backbones** | STGNN, DCRNN, ASTGNN, TGCN | `bash scripts/baselines_pems_run.sh` |
| **Retrieval / continual (STGNN backbone)** | PECPM, STRAP | `bash scripts/baselines_pems_run.sh` |
| **Test-time calibration** | ST-TTC | `bash scripts/sttc_run.sh` |
| **Other static backbones** | GWN, STID, iTransformer, DLinear, STNorm, STAEformer | `bash scripts/extra_baselines_run.sh` |
| **Ours** | A2TTA | `scripts/a2tta_lite_*` (see above) |

`scripts/pemsXX_run.sh` runs the **full per-dataset pipeline** end-to-end (Retrain →
auto-link → Pretrain → Online-NN → Online-AN → TrafficStream → STKEC → EAC) and is the
recommended starting point, because it also produces the Online-AN checkpoints A2TTA needs.

**Examples**

```bash
# Full pipeline on PEMS05 (produces naïve + continual baselines + Online-AN ckpts)
bash scripts/pems05_run.sh

# STRAP-paper backbones + PECPM + STRAP, just two datasets, on GPU 0
DATASETS="PEMS04 PEMS05" GPU=0 bash scripts/baselines_pems_run.sh

# Extra static backbones, only GWN + STID, all datasets
METHODS="gwn stid" bash scripts/extra_baselines_run.sh

# ST-TTC on a subset
DATASETS="pems05 pems06" GPU=0 bash scripts/sttc_run.sh

# Run a single method directly via main.py
python main.py --conf conf/PEMS05/eac.json --gpuid 0 --seed 51
```

Per-year metrics for every baseline are written to
`log/<DATASET>/<logname>-<seed>/<logname>.log`.

---

## 🙏 Acknowledgements

This benchmark builds on the data and code of several prior works, which we gratefully
acknowledge:

- **TrafficStream** (IJCAI'23) — [paper](https://arxiv.org/abs/2106.06273) · [repo](https://github.com/AprLie/TrafficStream)
- **EAC** (ICLR'25) — [paper](https://openreview.net/pdf?id=FRzCIlkM7I) · [code](https://github.com/Onedean/EAC)
- **STKEC** (TITS'23) — [paper](https://ieeexplore.ieee.org/document/10101714/) · [repo](https://github.com/wangbinwu13116175205/STKEC)
- **ST-TTC** (NeurIPS'25) — [paper](https://arxiv.org/pdf/2506.00635) · [repo](https://github.com/Onedean/ST-TTC)
- **STRAP** (NeurIPS'25), [paper](https://arxiv.org/abs/2505.19547/) · [repo](https://github.com/HoweyZ/STRAP)
- **PECPM** (KDD'23), and the conventional backbones GWN / STID /
  iTransformer / DLinear / ST-Norm / STAEformer.

## License

Released under the Apache-2.0 License — see [`LICENSE`](./LICENSE).
