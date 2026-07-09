# Figures & notebooks

Reproducible figures for the paper. **Each notebook embeds its rendered PNG**, so the figures
show on open **without re-running**. Re-running the render cells regenerates the `.png`/`.pdf`
(400 dpi PNG + true-vector PDF).

Every notebook is a thin wrapper around a `render_*.py` module whose top holds a
`STYLE / LAYOUT KNOBS` block — edit a knob in the notebook's KNOBS cell, re-run the render
cell, done. (Do **not** `importlib.reload` inside the render cell; that resets the knobs.)

| Notebook | Render module | Output |
|---|---|---|
| `hyper-a2tta.ipynb` | `render_hyper_a2tta.py` | `hyper_a2tta.{png,pdf}` — HP-sensitivity (OAT, 6 knobs × 2 backbones) |
| `new_sensor_baselines.ipynb` | `render_new_sensor_baselines.py` | `new_sensor_baselines{,2}.{png,pdf}` |
| `per_horizon_lines.ipynb` | `render_per_horizon.py` | `per_horizon_{all,new}_pems{3478,9}.{png,pdf}` |
| — (script only) | `render_final.py` (+ `_cohort_lib.py`) | `pems_grid_final_{carto,vector}.{png,pdf}` — sensor maps |
| — (static) | — | `fig_churn.{png,pdf}` — yearly sensor churn |
| — (script only) | `render_result_tables.py` | renders `../tables/*.md` → zoomable PDF |

## Data needed to regenerate

The figures are computed from the evaluation summaries under `run_logs/` (released with the
datasets, see [`../data/README.md`](../data/README.md)). After dropping `run_logs/` at the
**repo root**, the render modules find their inputs automatically:

- `render_hyper_a2tta.py` → `run_logs/hpAna_*/csv/*.csv`
- `render_per_horizon.py` / `render_new_sensor_baselines.py` → `run_logs/eval_new_sensors_*/summary.csv`
  (+ `../tables/*.tex` for labels)
- `render_final.py` (sensor maps) additionally needs per-sensor coordinate metadata and
  `cartopy` (basemap tiles are fetched online); it is a one-off geographic figure.

## Method spec

[`ours_method_spec.md`](ours_method_spec.md) is the faithful, code-grounded description of the
A2TTA method (frozen backbone → FiLM calibrator → delayed-label online TTA → discardable local
clone), backbone-agnostic, with a suggested figure layout — used to draft the overview figure.
