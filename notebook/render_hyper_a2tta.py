#!/usr/bin/env python3
"""
HP-sensitivity figure for **Ours (A2TTA)** on BOTH backbones (OL-AN &
STAEformer). One-at-a-time (OAT) sweep of six knobs around the default config
`lr1e-3 / steps3 / pool512 / budget0.25 / warmup3 / local3`.

Data source
-----------
    run_logs/hpAna_*/csv/HPA_<bb>_<DSID>__<knob>__<val>__s<seed>.csv
    run_logs/hpAna_*/csv/HPA_<bb>_<DSID>__anchor__s<seed>.csv

Each CSV carries one `horizon=="Avg"` row PER YEAR; the per-seed scalar is the
mean of Avg-MAE over all years (identical to how the main-table / sweep
aggregate). For every knob we normalise each (dataset,backbone)'s Avg-MAE by its
own ANCHOR run -> %DeltaMAE, then average over the 3 datasets (PEMS03/05/06).
The single anchor run supplies the default-value point of all six curves, so
that point is 0 by construction.

Figure: 2x3 panels (one per knob). Two lines = the two backbones. Shaded band =
+/- std across the 3 datasets. Dotted vertical = the default value.

Outputs (figs_paper/): hyper_a2tta.{png,pdf}

# ======================================================================== #
# STYLE / LAYOUT KNOBS  — edit here, or override from the notebook before
# calling build(), e.g.  `rh.PANEL_H_IN = 2.4; rh.LEGEND_FS = 11`.
#   *_FS = font points     *_IN = inches
# ======================================================================== #
"""
import csv
import glob
import os.path as osp
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- panel geometry -------------------------------------------------------
PANEL_W_IN = 2.55       # width  of each subplot (inches)
PANEL_H_IN = 2.15       # height of each subplot (inches)
WSPACE = 0.34           # horizontal gap between panels (fraction of panel width)
HSPACE = 0.52           # vertical gap between panels (room for titles/x-labels)
LEGEND_STRIP_IN = 0.62  # top strip reserved for the legend row (shrink -> closer)

# --- fonts ----------------------------------------------------------------
TITLE_FS = 11.0         # panel title (knob name)
LABEL_FS = 10.0         # axis labels
TICK_FS = 8.5           # tick labels
LEGEND_FS = 10.5        # legend font
ANNOT_FS = 8.0          # in-panel annotations (worst-case %)

# --- lines / markers ------------------------------------------------------
LINE_LW = 1.9           # curve line width
MARKER_MS = 6.0         # marker size
BAND_ALPHA = 0.16       # shaded std-band opacity
DEFAULT_LW = 1.1        # dotted "default value" vertical line
ZERO_LW = 0.9           # dashed y=0 reference line
ANNOTATE_WORST = True   # print the worst-case %Delta near the top of each panel

DPI_SAVE = 400          # PNG dpi (PDF is true-vector)

# --- backbones: label, colour (Okabe-Ito), marker -------------------------
BACKBONES = [
    ("an",   "Ours (OL-AN)",       dict(color="#0072B2", marker="o")),   # blue
    ("stae", "Ours (STAEformer)",  dict(color="#D55E00", marker="s")),   # vermillion
]

# --- knobs: (tag, pretty label, ordered values incl anchor, anchor, xscale) --
HP_SPEC = [
    ("lr",     r"TTA learning rate $\eta$",         [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 1e-3, "log"),
    ("st",     "TTA steps / batch",                 [1, 2, 3, 5],                    3,    "linear"),
    ("pool",   r"delayed-label pool $|\mathcal{P}|$", [128, 256, 512, 1024],         512,  "log"),
    ("budget", r"selection budget $\rho$",           [0.1, 0.25, 0.5, 1.0],          0.25, "linear"),
    ("wu",     "calibrator warm-up (epochs)",       [1, 3, 5],                       3,    "linear"),
    ("local",  "local-clone steps",                 [1, 3, 5],                       3,    "linear"),
]
METRIC = "MAE"          # normalise on Avg-MAE
NCOLS = 3

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
CSV_GLOB = osp.join(REPO, "run_logs", "hpAna_*", "csv", "*.csv")


# ----------------------------------------------------------------------- #
#  data                                                                    #
# ----------------------------------------------------------------------- #
def _avg_metric(path):
    """per-seed scalar = mean of the per-year Avg-MAE rows."""
    vals = [float(r[METRIC]) for r in csv.DictReader(open(path))
            if r.get("horizon") == "Avg" and r.get(METRIC)]
    return st.mean(vals) if vals else None


def load(csv_glob=None):
    """-> raw[(bb, dsid, knob, valfloat|None)] = list over seeds of Avg-MAE."""
    csv_glob = csv_glob or CSV_GLOB
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"no HP-analysis CSVs at {csv_glob}")
    raw = defaultdict(list)
    for f in files:
        parts = osp.basename(f)[:-4].split("__")
        pre = parts[0].split("_")                      # HPA, bb, dsid
        if len(pre) < 3 or pre[0] != "HPA":
            continue
        bb, dsid = pre[1], pre[2]
        v = _avg_metric(f)
        if v is None:
            continue
        if parts[1] == "anchor":
            raw[(bb, dsid, "anchor", None)].append(v)
        else:
            raw[(bb, dsid, parts[1], float(parts[2]))].append(v)
    return raw


def curves(raw):
    """-> curve[(bb, knob, val)] = (mean %d, std %d, n_datasets)."""
    dsids = sorted({k[1] for k in raw})
    anchor = {(bb, d): st.mean(raw[(bb, d, "anchor", None)])
              for bb, _l, _s in BACKBONES for d in dsids
              if raw.get((bb, d, "anchor", None))}
    out = {}
    for bb, _l, _s in BACKBONES:
        for knob, _lbl, vals, anc, _sc in HP_SPEC:
            for val in vals:
                pcts = []
                for d in dsids:
                    a = anchor.get((bb, d))
                    if a is None:
                        continue
                    if abs(val - anc) < 1e-12:
                        m = a
                    else:
                        seeds = raw.get((bb, d, knob, float(val)))
                        if not seeds:
                            continue
                        m = st.mean(seeds)
                    pcts.append((m / a - 1.0) * 100.0)
                if pcts:
                    out[(bb, knob, val)] = (st.mean(pcts),
                                            st.pstdev(pcts) if len(pcts) > 1 else 0.0,
                                            len(pcts))
    return out, anchor, dsids


# ----------------------------------------------------------------------- #
#  figure                                                                  #
# ----------------------------------------------------------------------- #
def build(out_stem="hyper_a2tta", csv_glob=None, show_title=False):
    raw = load(csv_glob)
    curve, anchor, dsids = curves(raw)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })

    nrows = (len(HP_SPEC) + NCOLS - 1) // NCOLS
    fig_w = NCOLS * PANEL_W_IN * (1 + WSPACE)
    fig_h = nrows * PANEL_H_IN * (1 + HSPACE) + LEGEND_STRIP_IN
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE,
                        top=1 - LEGEND_STRIP_IN / fig_h,
                        left=0.085, right=0.985, bottom=0.09)

    for idx, (knob, lbl, vals, anc, sc) in enumerate(HP_SPEC):
        ax = axes[idx // NCOLS][idx % NCOLS]
        worst = 0.0
        for bb, _label, style in BACKBONES:
            xs, ys, es = [], [], []
            for val in vals:
                if (bb, knob, val) in curve:
                    mu, sd, _ = curve[(bb, knob, val)]
                    xs.append(val); ys.append(mu); es.append(sd)
                    worst = max(worst, abs(mu) + sd)
            if not xs:
                continue
            ax.plot(xs, ys, lw=LINE_LW, ms=MARKER_MS, mfc=style["color"],
                    mec="white", mew=0.6, **style, zorder=5)
            ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=style["color"], alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.axhline(0, color="0.45", ls="--", lw=ZERO_LW, zorder=1)
        ax.axvline(anc, color="#1B7837", ls=":", lw=DEFAULT_LW, alpha=0.7, zorder=1)
        ax.set_xscale(sc)
        ax.set_xticks(vals)
        ax.set_xticklabels([_fmt_val(v) for v in vals], fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_title(lbl, fontsize=TITLE_FS, pad=5)
        if idx % NCOLS == 0:
            ax.set_ylabel(r"$\Delta$Avg-MAE vs default (%)", fontsize=LABEL_FS)
        ax.grid(alpha=0.25, lw=0.6)
        ax.margins(x=0.08)
        if ANNOTATE_WORST and worst:
            ax.annotate(f"max |Δ| ≈ {worst:.1f}%", xy=(0.96, 0.94),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=ANNOT_FS, color="0.35")

    handles = [Line2D([0], [0], color=s["color"], marker=s["marker"], lw=LINE_LW,
                      ms=MARKER_MS, mfc=s["color"], mec="white", label=lab)
               for _bb, lab, s in BACKBONES]
    handles.append(Line2D([0], [0], color="#1B7837", ls=":", lw=DEFAULT_LW,
                          label="default value"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=LEGEND_FS, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.8, handletextpad=0.5)
    if show_title:
        fig.suptitle("A2TTA hyper-parameter sensitivity (OAT; mean over "
                     "PEMS03/05/06, seeds 51-53)", fontsize=TITLE_FS, y=1.02)

    png = osp.join(HERE, out_stem + ".png")
    pdf = osp.join(HERE, out_stem + ".pdf")
    fig.savefig(png, dpi=DPI_SAVE, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[hyper_a2tta] datasets={dsids}  anchor(OL-AN)="
          f"{ {d: round(anchor.get(('an', d), float('nan')), 2) for d in dsids} }")
    print(f"[hyper_a2tta] wrote {png}\n[hyper_a2tta] wrote {pdf}")
    return png, pdf


def _fmt_val(v):
    if isinstance(v, float) and v < 0.01:
        return f"{v:.0e}".replace("e-0", "e-")
    if v == int(v):
        return str(int(v))
    return str(v)


def main():
    build()


if __name__ == "__main__":
    main()
