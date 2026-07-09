#!/usr/bin/env python3
"""Aggregate the HP-sensitivity OAT sweep (scripts/submit_hpanalysis.sh) and
produce (1) a sensitivity figure and (2) a markdown/LaTeX table.

Design: for each of 6 knobs we vary it one-at-a-time around the anchor
(lr1e-3/st3/pool512/budget0.25/wu3/local3). The single ANCHOR job per
(backbone,dataset,seed) supplies the default-value point of ALL 6 curves.
We normalise each (dataset,backbone)'s Avg-MAE by its own anchor -> %DeltaMAE,
then average over the 3 datasets. Figure = 6 subplots, one line per backbone.

Usage: python scripts/hpanalysis_plot.py [ROOT]   (default: latest hpAna_* root)
Outputs (../tables/): hp_sensitivity.pdf/.png , hp_sensitivity.md
"""
import csv, glob, os, sys, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.dirname(HERE))

ROOT = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("run_logs/hpAna_*"))[-1]
CSVDIR = os.path.join(ROOT, "csv")
OUT = "../tables"

# knob -> (pretty label, ordered values incl anchor, anchor value, xscale)
HP_SPEC = [
    ("lr",     r"adapt_lr ($\eta$)",       [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 1e-3, "log"),
    ("st",     "adapt_steps",              [1, 2, 3, 5],                    3,    "linear"),
    ("pool",   "candidate_pool_size",      [128, 256, 512, 1024],           512,  "log"),
    ("budget", "budget_frac",              [0.1, 0.25, 0.5, 1.0],           0.25, "linear"),
    ("wu",     "warmup_epochs",            [1, 3, 5],                       3,    "linear"),
    ("local",  "local_steps",              [1, 3, 5],                       3,    "linear"),
]
BB_LABEL = {"an": "Ours (OL-AN)", "stae": "Ours (STAEformer)"}
BB_ORDER = ["an", "stae"]
METRIC = "MAE"   # normalise on Avg-MAE


def avg_metric(path):
    # CSV has one horizon=="Avg" row PER YEAR; the per-seed scalar is the mean
    # over all years (identical to how the sweep/main-table aggregate).
    vals = [float(r[METRIC]) for r in csv.DictReader(open(path))
            if r.get("horizon") == "Avg" and r.get(METRIC)]
    return st.mean(vals) if vals else None


def load():
    # raw[(bb,dsid,hp,valfloat_or_None)] = list over seeds of Avg-MAE
    raw = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CSVDIR, "*.csv"))):
        tag = os.path.basename(f)[:-4]
        parts = tag.split("__")
        pre = parts[0].split("_")            # HPA, bb, dsid
        if len(pre) < 3 or pre[0] != "HPA":
            continue
        bb, dsid = pre[1], pre[2]
        v = avg_metric(f)
        if v is None:
            continue
        if parts[1] == "anchor":
            raw[(bb, dsid, "anchor", None)].append(v)
        else:
            raw[(bb, dsid, parts[1], float(parts[2]))].append(v)
    return raw


def main():
    raw = load()
    n = sum(len(v) for v in raw.values())
    print(f"[hpAna] ROOT={ROOT}  csv-cells={len(raw)}  total-seed-runs={n}")
    dsids = sorted({k[1] for k in raw})
    # anchor mean-MAE per (bb,dsid)
    anchor = {(bb, d): st.mean(raw[(bb, d, "anchor", None)])
              for bb in BB_ORDER for d in dsids if raw.get((bb, d, "anchor", None))}

    # curve[(bb,hp)][val] = (mean %d over datasets, std %d over datasets, n_ds)
    curve = {}
    for bb in BB_ORDER:
        for hp, _lbl, vals, anc, _sc in HP_SPEC:
            for val in vals:
                pcts = []
                for d in dsids:
                    a = anchor.get((bb, d))
                    if a is None:
                        continue
                    if abs(val - anc) < 1e-12:
                        m = a                       # anchor reused as default point
                    else:
                        seeds = raw.get((bb, d, hp, float(val)))
                        if not seeds:
                            continue
                        m = st.mean(seeds)
                    pcts.append((m / a - 1.0) * 100.0)
                if pcts:
                    curve[(bb, hp, val)] = (st.mean(pcts),
                                            st.pstdev(pcts) if len(pcts) > 1 else 0.0,
                                            len(pcts))

    # ---------------- figure ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
        colors = {"an": "#1f77b4", "stae": "#d62728"}
        markers = {"an": "o", "stae": "s"}
        for ax, (hp, lbl, vals, anc, sc) in zip(axes.flat, HP_SPEC):
            for bb in BB_ORDER:
                xs, ys, es = [], [], []
                for val in vals:
                    if (bb, hp, val) in curve:
                        mu, sd, _ = curve[(bb, hp, val)]
                        xs.append(val); ys.append(mu); es.append(sd)
                if not xs:
                    continue
                ax.plot(xs, ys, marker=markers[bb], color=colors[bb],
                        label=BB_LABEL[bb], lw=1.8, ms=6)
                ax.fill_between(xs, [y - e for y, e in zip(ys, es)],
                                [y + e for y, e in zip(ys, es)],
                                color=colors[bb], alpha=0.15)
            ax.axhline(0, color="gray", ls="--", lw=0.8)
            ax.axvline(anc, color="green", ls=":", lw=1.0, alpha=0.6)
            ax.set_xscale(sc)
            ax.set_xticks(vals); ax.set_xticklabels([str(v) for v in vals], fontsize=8)
            ax.set_xlabel(lbl); ax.set_ylabel(r"$\Delta$Avg-MAE vs default (%)")
            ax.grid(alpha=0.25)
        axes.flat[0].legend(fontsize=9, loc="best")
        fig.suptitle("Ours HP sensitivity (OAT, mean over PEMS03/05/06, 3 seeds; "
                     "green dotted = default)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"hp_sensitivity.{ext}"), dpi=160)
        print(f"[hpAna] wrote {OUT}/hp_sensitivity.pdf/.png")
    except Exception as e:
        print(f"[hpAna] figure skipped: {e}")

    # ---------------- markdown table ----------------
    lines = ["# HP sensitivity (Ours, OAT)\n",
             f"Root `{ROOT}`. Datasets PEMS03/05/06, seeds 51-53, both backbones. "
             "Cells = mean %ΔAvg-MAE vs default over the 3 datasets (± std). "
             "Default point (·) = 0 by construction.\n"]
    for bb in BB_ORDER:
        lines.append(f"\n## {BB_LABEL[bb]}\n")
        for hp, lbl, vals, anc, _sc in HP_SPEC:
            cells = []
            for val in vals:
                if (bb, hp, val) in curve:
                    mu, sd, nd = curve[(bb, hp, val)]
                    tag = f"**{mu:+.2f}±{sd:.2f}**" if abs(val - anc) < 1e-12 else f"{mu:+.2f}±{sd:.2f}"
                    cells.append(f"{val}={tag}")
                else:
                    cells.append(f"{val}=—")
            lines.append(f"- **{lbl}**: " + " · ".join(cells))
    # anchor absolute reference
    lines.append("\n## Anchor absolute Avg-MAE (default config)\n")
    lines.append("| backbone | " + " | ".join(dsids) + " |")
    lines.append("|" + "---|" * (len(dsids) + 1))
    for bb in BB_ORDER:
        row = [BB_LABEL[bb]] + [f"{anchor.get((bb,d),float('nan')):.2f}" for d in dsids]
        lines.append("| " + " | ".join(row) + " |")
    open(os.path.join(OUT, "hp_sensitivity.md"), "w").write("\n".join(lines) + "\n")
    print(f"[hpAna] wrote {OUT}/hp_sensitivity.md")


if __name__ == "__main__":
    main()
