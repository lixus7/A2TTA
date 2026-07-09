#!/usr/bin/env python3
"""
Per-horizon prediction-error line figures (style mirrors STWave Fig. 8:
"Prediction for each time slice"), for **Ours (A2TTA) vs ST-TTC, EAC, GWN,
STNorm**.

Data source
-----------
The numbers come straight from the new-sensor evaluation summaries that
`make_new_sensor_tables.py` also consumes:

    run_logs/eval_new_sensors_*/summary.csv      (baselines)
    run_logs/eval_new_sensors_a2tta_*/summary.csv (Ours)

Each row is (dataset, method, seed, scope, horizon, year_count, MAE, RMSE,
MAPE). We replicate the table builder's aggregation: merge all summaries, dedup
by (dataset, method, horizon, seed) latest-wins, filter to one `scope`
("all"/"new"), mean over seeds (here one seed per method).

Horizons
--------
Each summary carries, per scope/year: cumulative "1".."12" (error averaged over
the first h steps; "3"/"6"/"12" reproduce the table), per-step "step1".."step12"
(error at exactly horizon h — the Fig. 8 convention), and "Avg". `HORIZON_MODE`:
"step" -> per-step (default look); "cum" -> cumulative; "auto" -> step if present.

Outputs (figs_paper/): per_horizon_{all,new}_pems{3478,9}.{png,pdf}

# ======================================================================== #
# STYLE / LAYOUT KNOBS  — edit here, or override from the notebook before
# calling render(), e.g.  `rph.LEGEND_FS = 11; rph.PANEL_H_IN = 2.2`.
#   *_FS  = font size in points        *_IN = size in inches
#   legend sits in a reserved strip at the top; shrink LEGEND_STRIP_IN to pull
#   it closer to the panels (that was the "too much whitespace" gap).
# ======================================================================== #
"""
import csv
import glob
import math
import os.path as osp
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# --- panel geometry -------------------------------------------------------
PANEL_W_IN = 2.05      # width  of each subplot (inches)
PANEL_H_IN = 2.00      # height of each subplot (inches)
WSPACE = 0.42          # horizontal gap between panels (fraction of panel width)
HSPACE = 0.55          # vertical gap between panels (room for titles/x-labels)

# --- legend ---------------------------------------------------------------
LEGEND_STRIP_IN = 0.34 # vertical strip reserved at the very top for the legend.
                       # SMALLER -> legend closer to the panels (less whitespace).
LEGEND_FS = 9.0        # legend font size
LEGEND_NCOL = 5        # entries per legend row (5 methods -> single row)
LEGEND_Y_PAD = 0.0     # extra figure-fraction lift above the panels (0 = flush)

# --- fonts ----------------------------------------------------------------
TITLE_FS = 8.2         # per-panel title, e.g. "(a) PEMS03 MAE"
YLABEL_FS = 7.8        # y-axis label (MAE / RMSE / MAPE)
XLABEL_FS = 7.6        # x-axis label ("Horizon")
TICK_FS = 6.8          # tick numbers

# --- save -----------------------------------------------------------------
DPI_SAVE = 400

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
SUMMARY_GLOB = osp.join(REPO, "run_logs", "eval_new_sensors_*", "summary.csv")

# --- Methods to plot: display label -> slug in summary.csv (order = legend) ---
METHODS = [
    ("Ours",   "a2tta_lite"),
    ("ST-TTC", "sttc"),
    ("EAC",    "eac"),
    ("GWN",    "retrain_gwn"),
    ("STNorm", "retrain_stnorm"),
]
# Okabe-Ito colour-blind-safe palette; Ours = vermillion + diamond + thick line
# so it pops exactly like the red STWave curve in the reference figure.
STYLE = {
    "Ours":   dict(color="#D55E00", marker="D", lw=2.1, ms=4.5, zorder=6),
    "ST-TTC": dict(color="#0072B2", marker="o", lw=1.3, ms=3.6, zorder=4),
    "EAC":    dict(color="#009E73", marker="^", lw=1.3, ms=3.8, zorder=4),
    "GWN":    dict(color="#CC79A7", marker="s", lw=1.3, ms=3.4, zorder=4),
    "STNorm": dict(color="#E69F00", marker="v", lw=1.3, ms=3.8, zorder=4),
}
METRICS = ["MAE", "RMSE", "MAPE"]
PEMS9 = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07", "PEMS08",
         "PEMS10", "PEMS11", "PEMS12"]
PEMS3478 = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]


# --------------------------------------------------------------------------- #
# Load + aggregate (mirrors eac/scripts/make_new_sensor_tables.py)
# --------------------------------------------------------------------------- #
def load_series(scope):
    """series[(ds, label, metric)] = {horizon_key: mean_over_seeds}."""
    slug2label = {slug: lbl for lbl, slug in METHODS}
    raw = {}  # (ds, slug, horizon, seed) -> {metric: val}
    for path in sorted(glob.glob(SUMMARY_GLOB)):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["scope"] != scope or r["method"] not in slug2label:
                    continue
                k = (r["dataset"], r["method"], r["horizon"], r["seed"])
                try:
                    raw[k] = {m: float(r[m]) for m in METRICS}
                except (KeyError, ValueError):
                    continue

    bucket = defaultdict(list)
    for (ds, slug, horizon, _seed), vals in raw.items():
        lbl = slug2label[slug]
        for m in METRICS:
            bucket[(ds, lbl, m, horizon)].append(vals[m])

    series = defaultdict(dict)
    for (ds, lbl, m, horizon), vals in bucket.items():
        series[(ds, lbl, m)][horizon] = float(np.mean(vals))
    return series


def horizon_axis(series, mode):
    """Return (mode_used, [(horizon_key, x_int), ...]) for the requested mode."""
    keys = set()
    for hmap in series.values():
        keys.update(hmap)
    step = sorted((k for k in keys if k.startswith("step")), key=lambda k: int(k[4:]))
    cum = sorted((k for k in keys if k.isdigit()), key=int)
    if mode == "step" or (mode == "auto" and step):
        return "step", [(k, int(k[4:])) for k in step]
    return "cum", [(k, int(k)) for k in cum]


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def _set_rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def render(scope, datasets, datasets_per_row, out_stem, horizon_mode="auto",
           panel_w_in=None, panel_h_in=None):
    """One panel per (dataset, metric). `datasets_per_row` datasets are packed
    side by side, so total columns = datasets_per_row * 3 (the 3 metrics).

    panel_w_in / panel_h_in override the module PANEL_*_IN for THIS figure only
    (e.g. a shorter panel height for the tall 1-dataset-per-row 3478 layout)."""
    _set_rc()
    pw = PANEL_W_IN if panel_w_in is None else panel_w_in
    ph = PANEL_H_IN if panel_h_in is None else panel_h_in
    series = load_series(scope)
    mode, axis = horizon_axis(series, horizon_mode)
    if not axis:
        print(f"[skip] {out_stem}: no horizons found for scope={scope}")
        return
    hkeys = [k for k, _x in axis]
    xs = [x for _k, x in axis]

    present = [d for d in datasets if any((d, lbl, "MAE") in series
                                          for lbl, _s in METHODS)]
    if not present:
        print(f"[skip] {out_stem}: none of {datasets} present for scope={scope}")
        return

    n_ds = len(present)
    nrows = math.ceil(n_ds / datasets_per_row)
    ncols = datasets_per_row * len(METRICS)

    fig_w = pw * ncols
    fig_h = ph * nrows + LEGEND_STRIP_IN
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             dpi=150, squeeze=False)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)

    panel_letter = iter("abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    used = set()
    for di, ds in enumerate(present):
        ds_r = di // datasets_per_row
        ds_c = di % datasets_per_row
        for mi, metric in enumerate(METRICS):
            ax = axes[ds_r][ds_c * len(METRICS) + mi]
            used.add((ds_r, ds_c * len(METRICS) + mi))
            for lbl, _slug in METHODS:
                hmap = series.get((ds, lbl, metric), {})
                y = [hmap.get(k, np.nan) for k in hkeys]
                if np.all(np.isnan(y)):
                    continue
                st = STYLE[lbl]
                ax.plot(xs, y, label=lbl, color=st["color"], marker=st["marker"],
                        linewidth=st["lw"], markersize=st["ms"],
                        markeredgewidth=0.4, markeredgecolor="white",
                        zorder=st["zorder"], clip_on=False)

            ax.grid(True, color="0.88", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=TICK_FS, length=2)
            if len(xs) <= 12:
                ax.set_xticks(xs)
            ax.margins(x=0.02)

            unit = " (%)" if metric == "MAPE" else ""
            tag = next(panel_letter)
            ax.set_title(f"({tag}) {ds} {metric}{unit}", fontsize=TITLE_FS, pad=3)
            ax.set_xlabel("Horizon", fontsize=XLABEL_FS, labelpad=1.5)
            if mi == 0:
                ax.set_ylabel(metric + unit, fontsize=YLABEL_FS, labelpad=2)

    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r][c].set_visible(False)

    # pack panels into everything below `top`, leaving the LEGEND_STRIP at top.
    top = 1.0 - LEGEND_STRIP_IN / fig_h
    fig.tight_layout(rect=[0, 0, 1, top])

    # legend sits with its lower edge flush at `top` (+ optional lift) so the
    # gap to the panels is exactly LEGEND_STRIP_IN (minus legend height).
    handles = [Line2D([0], [0], color=STYLE[lbl]["color"], marker=STYLE[lbl]["marker"],
                      linewidth=STYLE[lbl]["lw"], markersize=STYLE[lbl]["ms"],
                      markeredgecolor="white", markeredgewidth=0.4, label=lbl)
               for lbl, _s in METHODS]
    leg = fig.legend(handles=handles, ncol=LEGEND_NCOL, loc="lower center",
                     bbox_to_anchor=(0.5, top + LEGEND_Y_PAD), frameon=False,
                     fontsize=LEGEND_FS, handlelength=2.0, columnspacing=1.8,
                     handletextpad=0.5)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_color(STYLE["Ours"]["color"])
            t.set_fontweight("bold")

    png = osp.join(HERE, out_stem + ".png")
    pdf = osp.join(HERE, out_stem + ".pdf")
    fig.savefig(png, dpi=DPI_SAVE, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_stem}: scope={scope}, {n_ds} ds, {nrows}x{ncols} grid, "
          f"x={mode} horizons {xs[0]}..{xs[-1]} -> {png}")


def main():
    # 3478 -> 3 columns (1 dataset/row), shorter panels so it isn't too tall;
    # all-9 -> 9 columns (3 datasets/row).
    H3478 = 1.40  # panel height (in) for the 4-row 3478 figures — kept short
    render("all", PEMS3478, datasets_per_row=1, out_stem="per_horizon_all_pems3478", panel_h_in=H3478)
    render("all", PEMS9,    datasets_per_row=3, out_stem="per_horizon_all_pems9")
    render("new", PEMS3478, datasets_per_row=1, out_stem="per_horizon_new_pems3478", panel_h_in=H3478)
    render("new", PEMS9,    datasets_per_row=3, out_stem="per_horizon_new_pems9")


if __name__ == "__main__":
    main()
