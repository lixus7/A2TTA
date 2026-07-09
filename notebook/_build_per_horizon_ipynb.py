#!/usr/bin/env python3
"""Build per_horizon_lines.ipynb (valid nbformat-4, no nbformat dependency)
from render_per_horizon.py, embedding the rendered PNGs so the figures show on
open. Run with the stg kernel to regenerate."""
import base64
import json
import os.path as osp

HERE = osp.dirname(osp.abspath(__file__))

MD = """\
# Per-horizon prediction error — Ours vs ST-TTC / EAC / GWN / STNorm

Line figures in the style of STWave **Fig. 8** ("prediction for each time
slice"): one curve per method, error vs forecast horizon, a panel per
(dataset, metric).

- **Source:** `eac/run_logs/eval_new_sensors_*/summary.csv` — the *same*
  summaries the tables are built from (dedup by ds/method/horizon/seed
  latest-wins, mean over seeds; here one seed per method).
- **Methods / colours:** Ours (A2TTA) = vermillion diamonds, thick; ST-TTC,
  EAC, GWN, STNorm.
- **Scope:** `all` = every sensor; `new` = newly-added sensors only.
- **Layout:** total columns = `datasets_per_row * 3` (the 3 metrics). The
  PEMS03/04/07/08 figures use `datasets_per_row=1` → **3 columns** (one dataset
  per row, short panels via `H3478`); the all-9 figures use `datasets_per_row=3`
  → **9 columns**.
- **Tuning:** the cell below ("STYLE / LAYOUT KNOBS") sets every font size, the
  legend gap, panel size and spacing — edit and re-run. Defaults live at the top
  of `render_per_horizon.py`.
- **Outputs:** `per_horizon_{all,new}_pems{3478,9}.{png,pdf}` (400 dpi PNG +
  true-vector PDF).
"""

CELL_IMPORTS = '''\
%matplotlib inline
import os, sys, importlib
sys.path.insert(0, os.getcwd())          # notebook lives in figs_paper/
import render_per_horizon as rph
importlib.reload(rph)
from IPython.display import Image, display
print("methods:", [l for l, _ in rph.METHODS])
print("summary glob:", rph.SUMMARY_GLOB)
'''

CELL_KNOBS = '''\
# ============================ STYLE / LAYOUT KNOBS ============================
# Edit any of these, then re-run the render cells below. (Full list + comments
# at the top of render_per_horizon.py.)  *_FS = font points, *_IN = inches.
rph.PANEL_W_IN     = 2.05    # subplot width  (inches)
rph.PANEL_H_IN     = 2.00    # subplot height (inches)
rph.WSPACE         = 0.42    # horizontal gap between panels
rph.HSPACE         = 0.55    # vertical gap between panels (room for titles/x-labels)

rph.LEGEND_STRIP_IN = 0.34   # top strip reserved for the legend row.
                             #   -> SMALLER pulls the legend closer to the panels
                             #      (this is the "too much whitespace" control).
rph.LEGEND_FS      = 9.0     # legend font size
rph.LEGEND_NCOL    = 5       # legend entries per row (5 = single row)
rph.LEGEND_Y_PAD   = 0.0     # extra lift of the legend above the panels (fraction)

rph.TITLE_FS       = 8.2     # per-panel title, e.g. "(a) PEMS03 MAE"
rph.YLABEL_FS      = 7.8     # y-axis label (MAE / RMSE / MAPE)
rph.XLABEL_FS      = 7.6     # x-axis label ("Horizon")
rph.TICK_FS        = 6.8     # tick numbers

# Per-method line look (colour / marker / linewidth / markersize):
# rph.STYLE["Ours"]["lw"] = 2.4
# rph.STYLE["Ours"]["color"] = "#D55E00"

# Shorter panel height for the 1-dataset-per-row 3478 figures (the all-9 figures
# keep the default rph.PANEL_H_IN). Bump this up to make 3478 taller.
H3478 = 1.40

# Horizon x-axis: "step" per-step (Fig. 8) | "cum" cumulative 1..h | "auto"
HMODE = "auto"
'''

CELL_3478_ALL = '''\
# ---- PRIORITY: all-sensor, PEMS03/04/07/08 -> 3 columns (1 dataset / row) ----
rph.render("all", rph.PEMS3478, datasets_per_row=1,
           out_stem="per_horizon_all_pems3478", horizon_mode=HMODE, panel_h_in=H3478)
display(Image("per_horizon_all_pems3478.png"))
'''

CELL_9_ALL = '''\
# ---- all-sensor, all 9 datasets -> 9 columns (3 datasets / row) ----
rph.render("all", rph.PEMS9, datasets_per_row=3,
           out_stem="per_horizon_all_pems9", horizon_mode=HMODE)
display(Image("per_horizon_all_pems9.png"))
'''

CELL_3478_NEW = '''\
# ---- new-sensor, PEMS03/04/07/08 -> 3 columns (1 dataset / row) ----
rph.render("new", rph.PEMS3478, datasets_per_row=1,
           out_stem="per_horizon_new_pems3478", horizon_mode=HMODE, panel_h_in=H3478)
display(Image("per_horizon_new_pems3478.png"))
'''

CELL_9_NEW = '''\
# ---- new-sensor, all 9 datasets -> 9 columns (3 datasets / row) ----
rph.render("new", rph.PEMS9, datasets_per_row=3,
           out_stem="per_horizon_new_pems9", horizon_mode=HMODE)
display(Image("per_horizon_new_pems9.png"))
'''


def code_cell(src, outputs=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": outputs or [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def png_display(fname):
    b64 = base64.b64encode(open(osp.join(HERE, fname), "rb").read()).decode()
    return [{"output_type": "display_data",
             "data": {"image/png": b64, "text/plain": ["<IPython.core.display.Image object>"]},
             "metadata": {}}]


nb = {
    "cells": [
        md_cell(MD),
        code_cell(CELL_IMPORTS),
        code_cell(CELL_KNOBS),
        code_cell(CELL_3478_ALL, outputs=png_display("per_horizon_all_pems3478.png")),
        code_cell(CELL_9_ALL,    outputs=png_display("per_horizon_all_pems9.png")),
        code_cell(CELL_3478_NEW, outputs=png_display("per_horizon_new_pems3478.png")),
        code_cell(CELL_9_NEW,    outputs=png_display("per_horizon_new_pems9.png")),
    ],
    "metadata": {
        "kernelspec": {"display_name": "stg", "language": "python", "name": "stg"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = osp.join(HERE, "per_horizon_lines.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
