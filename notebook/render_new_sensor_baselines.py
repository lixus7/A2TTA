#!/usr/bin/env python3
"""
New-sensor baseline comparison figure (style mirrors the demo grouped-bar plot).

Reads tables/tsas_new_sensors_part1.tex, plots PEMS03 + PEMS04, all 3 metrics
(MAE / RMSE / MAPE) on the "Avg" horizon, as a 2-row x 3-col grid of grouped
bar charts. Bars are colored by baseline category; "Ours" (A2TTA) is red.

Outputs (figs_paper/): new_sensor_baselines.{png,pdf}
"""
import os
import os.path as osp
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
TEX = osp.join(REPO, "tables", "tsas_new_sensors_part1.tex")

# --- Column schema: (group, label) in the exact order of the .tex header ------
COLUMNS = [
    ("Static STGNN Backbones", "DCRNN"), ("Static STGNN Backbones", "ASTGNN"),
    ("Static STGNN Backbones", "TGCN"),  ("Static STGNN Backbones", "GWN"),
    ("Naïve Schemes", "Pretrain"), ("Naïve Schemes", "Retrain"),
    ("Naïve Schemes", "OL-NN"),    ("Naïve Schemes", "OL-AN"),
    ("Evolving-Graph Continual", "TrafStm"), ("Evolving-Graph Continual", "PECPM"),
    ("Evolving-Graph Continual", "STKEC"),   ("Evolving-Graph Continual", "EAC"),
    ("Retrieval / TTC", "STRAP"), ("Retrieval / TTC", "ST-TTC"),
    ("Other Static Backbones", "STID"), ("Other Static Backbones", "STNorm"),
    ("Other Static Backbones", "iTrans"), ("Other Static Backbones", "DLinear"),
    ("Ours", "A2TTA"),
]
LABELS = [lbl for _g, lbl in COLUMNS]
GROUPS = [g for g, _l in COLUMNS]

GROUP_ORDER = ["Static STGNN Backbones", "Naïve Schemes",
               "Evolving-Graph Continual", "Retrieval / TTC",
               "Other Static Backbones", "Ours"]
# Okabe-Ito colour-blind-safe categorical palette; "Ours" = vermillion (most
# distinct, further reinforced by bold edge + bold label + rightmost position).
GROUP_COLOR = {
    "Static STGNN Backbones":   "#0072B2",  # blue
    "Naïve Schemes":            "#56B4E9",  # sky blue
    "Evolving-Graph Continual": "#009E73",  # bluish green
    "Retrieval / TTC":          "#CC79A7",  # reddish purple
    "Other Static Backbones":   "#E69F00",  # orange
    "Ours":                     "#D55E00",  # vermillion
}
GROUP_LEGEND = {  # shorter legend labels, like the demo
    "Static STGNN Backbones": "Static STGNN",
    "Naïve Schemes": "Naïve",
    "Evolving-Graph Continual": "Evolving Continual",
    "Retrieval / TTC": "Retrieval / TTC",
    "Other Static Backbones": "Other Static",
    "Ours": "Ours",
}

DATASETS = ["PEMS03", "PEMS04"]
METRICS = ["MAE", "RMSE", "MAPE"]
HORIZON = "Avg"

# Tilt of the x-axis method names, in degrees (90 = vertical, 0 = horizontal).
XTICK_ROTATION = 45


# --------------------------------------------------------------------------- #
# Parse the LaTeX table -> data[(ds, metric, horizon)] = [mean per method]
# --------------------------------------------------------------------------- #
def parse_tex(path):
    txt = open(path).read()
    body = txt.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    data = {}
    cur_ds = cur_metric = None
    num = re.compile(r"(-?\d+\.\d+)")
    for frag in body.split(r"\\"):
        if "$_{\\pm" not in frag:
            continue
        # drop cmidrule prefixes that can ride along on a fragment
        frag = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", frag)
        cells = [c.strip() for c in frag.split("&")]
        if len(cells) < 3 + len(COLUMNS):
            continue
        meta, methodcells = cells[:3], cells[3:3 + len(COLUMNS)]
        m0 = re.search(r"PEMS\d+", meta[0])
        if m0:
            cur_ds = m0.group(0)
        if "MAE" in meta[1]:
            cur_metric = "MAE"
        elif "RMSE" in meta[1]:
            cur_metric = "RMSE"
        elif "MAPE" in meta[1]:
            cur_metric = "MAPE"
        horizon = meta[2]
        vals = []
        for c in methodcells:
            m = num.search(c)
            vals.append(float(m.group(1)) if m else np.nan)
        data[(cur_ds, cur_metric, horizon)] = vals
    return data


def main():
    data = parse_tex(TEX)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,   # embed TrueType -> selectable vector text
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    bar_colors = [GROUP_COLOR[g] for g in GROUPS]
    x = np.arange(len(LABELS))

    fig, axes = plt.subplots(
        len(DATASETS), len(METRICS),
        figsize=(7.1, 5.7), dpi=150,
    )

    for r, ds in enumerate(DATASETS):
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            vals = np.array(data[(ds, metric, HORIZON)], dtype=float)

            # Per-panel cap to the competitive range so good methods stay
            # distinguishable (the demo likewise scales each subplot
            # independently); extreme failures are clipped + hatched + labelled
            # with their true value.
            vmin = np.nanmin(vals)
            competitive = vals[vals <= 3.0 * vmin]
            cap = 1.18 * np.nanmax(competitive)
            disp = np.minimum(vals, cap)
            clip_mask = vals > cap + 1e-9

            bars = ax.bar(x, disp, width=0.74, color=bar_colors,
                          edgecolor="white", linewidth=0.4, zorder=3)
            # Clipped bars: hatch + dark edge so they don't read as honest heights.
            for b, clipped in zip(bars, clip_mask):
                if clipped:
                    b.set_hatch("////")
                    b.set_edgecolor("#3a3a3a")
                    b.set_linewidth(0.5)
            # Make the "Ours" bar pop.
            bars[-1].set_edgecolor("#7a3500")
            bars[-1].set_linewidth(0.9)

            ax.set_ylim(0, cap * 1.30)
            ax.set_xlim(-0.7, len(LABELS) - 0.3)

            # value labels (vertical) above each bar; 1 decimal everywhere,
            # trailing ↑ marks a clipped bar.
            for i, (b, v, clipped) in enumerate(zip(bars, vals, clip_mask)):
                txt = f"{v:.1f}↑" if clipped else f"{v:.1f}"
                is_ours = i == len(LABELS) - 1
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + cap * 0.02,
                        txt, ha="center", va="bottom", rotation=90,
                        fontsize=6.0, color=GROUP_COLOR["Ours"] if is_ours else "#2b2b2b",
                        fontweight="bold" if is_ours else "normal", zorder=4)

            ax.set_xticks(x)
            ax.set_xticklabels(LABELS, rotation=XTICK_ROTATION, fontsize=6.2,
                               ha=("center" if XTICK_ROTATION == 90 else "right"),
                               rotation_mode="anchor")
            for tl, g in zip(ax.get_xticklabels(), GROUPS):
                if g == "Ours":
                    tl.set_color(GROUP_COLOR["Ours"])
                    tl.set_fontweight("bold")
            ax.tick_params(axis="y", labelsize=7.0, length=2)
            ax.tick_params(axis="x", length=0, pad=1.5)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="0.85", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)

            unit = " (%)" if metric == "MAPE" else ""
            if c == 0:
                ax.set_ylabel(f"{ds}", fontsize=9.5, fontweight="bold", labelpad=5)
            if r == 0:
                ax.set_title(f"{metric}{unit}", fontsize=9.5, pad=4)

    # --- shared top legend (one entry per category) ---
    handles = [Patch(facecolor=GROUP_COLOR[g], edgecolor="white",
                     label=GROUP_LEGEND[g]) for g in GROUP_ORDER]
    leg = fig.legend(handles=handles, ncol=6, loc="upper center",
                     bbox_to_anchor=(0.5, 1.025), frameon=False,
                     fontsize=8.0, handlelength=1.1, handleheight=1.1,
                     columnspacing=1.4, handletextpad=0.4)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_color(GROUP_COLOR["Ours"])
            t.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    png = osp.join(HERE, "new_sensor_baselines.png")
    pdf = osp.join(HERE, "new_sensor_baselines.pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[ok] wrote {png}")
    print(f"[ok] wrote {pdf}")


def main_3x2():
    """3-row x 2-col variant: rows = metrics, cols = datasets. Saves *2.{png,pdf}."""
    data = parse_tex(TEX)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.7, "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    bar_colors = [GROUP_COLOR[g] for g in GROUPS]
    x = np.arange(len(LABELS))

    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(7.0, 8.8), dpi=150)
    for r, metric in enumerate(METRICS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            vals = np.array(data[(ds, metric, HORIZON)], dtype=float)
            vmin = np.nanmin(vals)
            cap = 1.18 * np.nanmax(vals[vals <= 3.0 * vmin])
            disp = np.minimum(vals, cap)
            clip_mask = vals > cap + 1e-9

            bars = ax.bar(x, disp, width=0.74, color=bar_colors,
                          edgecolor="white", linewidth=0.4, zorder=3)
            for b, clipped in zip(bars, clip_mask):
                if clipped:
                    b.set_hatch("////"); b.set_edgecolor("#3a3a3a"); b.set_linewidth(0.5)
            bars[-1].set_edgecolor("#7a3500"); bars[-1].set_linewidth(0.9)

            ax.set_ylim(0, cap * 1.30)
            ax.set_xlim(-0.7, len(LABELS) - 0.3)
            for i, (b, v, clipped) in enumerate(zip(bars, vals, clip_mask)):
                is_ours = i == len(LABELS) - 1
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + cap * 0.02,
                        f"{v:.1f}↑" if clipped else f"{v:.1f}",
                        ha="center", va="bottom", rotation=90, fontsize=6.0,
                        color=GROUP_COLOR["Ours"] if is_ours else "#2b2b2b",
                        fontweight="bold" if is_ours else "normal", zorder=4)

            ax.set_xticks(x)                           # every subplot shows the method names
            ax.set_xticklabels(LABELS, rotation=XTICK_ROTATION, fontsize=6.2,
                               ha=("center" if XTICK_ROTATION == 90 else "right"),
                               rotation_mode="anchor")
            for tl, g in zip(ax.get_xticklabels(), GROUPS):
                if g == "Ours":
                    tl.set_color(GROUP_COLOR["Ours"]); tl.set_fontweight("bold")
            ax.tick_params(axis="y", labelsize=7.0, length=2)
            ax.tick_params(axis="x", length=0, pad=1.5)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="0.85", linewidth=0.5, zorder=0); ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(metric + (" (%)" if metric == "MAPE" else ""),
                              fontsize=9.5, fontweight="bold", labelpad=5)
            if r == 0:
                ax.set_title(ds, fontsize=10, pad=4, fontweight="bold")

    handles = [Patch(facecolor=GROUP_COLOR[g], edgecolor="white",
                     label=GROUP_LEGEND[g]) for g in GROUP_ORDER]
    leg = fig.legend(handles=handles, ncol=6, loc="upper center",
                     bbox_to_anchor=(0.5, 1.012), frameon=False, fontsize=8.0,
                     handlelength=1.1, handleheight=1.1, columnspacing=1.4, handletextpad=0.4)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_color(GROUP_COLOR["Ours"]); t.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.965])

    png = osp.join(HERE, "new_sensor_baselines2.png")
    pdf = osp.join(HERE, "new_sensor_baselines2.pdf")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[ok] wrote {png}")
    print(f"[ok] wrote {pdf}")


if __name__ == "__main__":
    main()
    main_3x2()
