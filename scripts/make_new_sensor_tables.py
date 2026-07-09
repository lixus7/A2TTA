#!/usr/bin/env python3
"""
make_new_sensor_tables.py

Reads all `summary.csv` produced by `eval_new_sensors_run.sh` (under
run_logs/eval_new_sensors_*/) and emits Markdown + LaTeX tables of
*newly-added-sensor* MAE/RMSE/MAPE for horizon 3, 6, 12, Avg.

Layout matches tables/tsas_main_table_part{1,2}.tex:
  4 column groups: Static STGNN Backbones | Naive Schemes | Evolving-Graph Continual | Retrieval/TTC
  Per (dataset, metric, horizon) row, **bold** = best, \\underline{} = 2nd best.
  Two .tex files mirroring the part1/part2 dataset split.

Inputs : run_logs/eval_new_sensors_*/summary.csv  (scope=new only)
Outputs in <repo_root>/tables/:
  tsas_new_sensors_part1.tex   PEMS03/04/05/06/07
  tsas_new_sensors_part2.tex   PEMS08/10/11/12
  tsas_new_sensors.md          all datasets present, same column layout
"""
import csv
import glob
import os
import os.path as osp
import sys
from collections import defaultdict

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.dirname(HERE)
REPO = osp.dirname(ROOT)
TABLES_DIR = osp.join(REPO, "tables")

_ALL_SUMMARIES = sorted(glob.glob(osp.join(ROOT, "run_logs", "eval_new_sensors_*", "summary.csv")))
if not _ALL_SUMMARIES:
    print("[error] no run_logs/eval_new_sensors_*/summary.csv found", file=sys.stderr)
    sys.exit(1)
# Merge ALL summary.csv files, deduping by (dataset, method, seed): later runs
# override earlier ones for the same key. The folder names embed timestamps
# (eval_new_sensors_YYYYMMDD_HHMMSS), so sorted order == chronological order
# and the last write wins. This lets you run a single method (e.g. STKEC) in
# a fresh batch and have it merged into the existing 13-method table without
# losing or double-counting anything.
#
# Override with EVAL_SUMMARIES="path1 path2 ..." to restrict.
_ENV_OVERRIDE = os.environ.get("EVAL_SUMMARIES", "").strip()
if _ENV_OVERRIDE:
    SUMMARIES = _ENV_OVERRIDE.split()
else:
    SUMMARIES = _ALL_SUMMARIES
print(f"[*] merging {len(SUMMARIES)} summary csv(s) (dedup by ds/method/seed, latest wins):")
for p in SUMMARIES:
    print(f"      {p}")

# ---------------------------------------------------------------------------
# Column schema — mirrors tables/tsas_main_table_part1.tex order exactly.
# Each entry: (group, column_label, slug_in_summary_csv).
# ---------------------------------------------------------------------------
COLUMNS = [
    ("Static STGNN Backbones",  "DCRNN",          "retrain_dcrnn"),
    ("Static STGNN Backbones",  "ASTGNN",         "retrain_astgnn"),
    ("Static STGNN Backbones",  "TGCN",           "retrain_tgcn"),
    ("Static STGNN Backbones",  "GWN",            "retrain_gwn"),

    (r'Na\"ive Schemes',        "Pretrain",       "pretrain_st"),
    (r'Na\"ive Schemes',        "Retrain",        "retrain_st"),
    (r'Na\"ive Schemes',        "OL-NN",          "oneline_st_nn"),
    (r'Na\"ive Schemes',        "OL-AN",          "oneline_st_an"),

    ("Evolving-Graph Continual", "TrafStm",       "trafficstream"),
    ("Evolving-Graph Continual", "PECPM",         "pecpm"),
    ("Evolving-Graph Continual", "STKEC",         "stkec"),
    ("Evolving-Graph Continual", "EAC",           "eac"),

    ("Retrieval / TTC",         "STRAP",          "strap"),
    ("Retrieval / TTC",         "ST-TTC",         "sttc"),

    ("Other Static Backbones",  "STID",           "retrain_stid"),
    ("Other Static Backbones",  "STNorm",         "retrain_stnorm"),
    ("Other Static Backbones",  "iTrans",         "retrain_itransformer"),
    ("Other Static Backbones",  "DLinear",        "retrain_dlinear"),

    ("Ours",                    "a2tta-olan",     "a2tta_lite"),
    ("Ours",                    "a2tta-staef",    "a2tta_staef"),
]

DATASET_PART1 = ["PEMS03", "PEMS04", "PEMS05", "PEMS06", "PEMS07"]
DATASET_PART2 = ["PEMS08", "PEMS10", "PEMS11", "PEMS12", "TFNSW"]
HORIZONS = ["3", "6", "12", "Avg"]
METRICS = ["MAE", "RMSE", "MAPE"]
SCOPE = "new"

# Lower-is-better for all three.
# ---------------------------------------------------------------------------

# Dedup by (dataset, method, horizon, seed): later summary overrides earlier.
# Each summary.csv is processed in chronological order (SUMMARIES is sorted by
# timestamped folder name), so the final value per (ds, m, h, seed) is the
# most recent re-evaluation of that exact run.
# raw[(ds, method, horizon, seed)] = {"MAE": ..., "RMSE": ..., "MAPE": ...}
raw = {}
for path in SUMMARIES:
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["scope"] != SCOPE:
                continue
            k = (r["dataset"], r["method"], r["horizon"], r["seed"])
            raw[k] = {m: float(r[m]) for m in METRICS}

# Fold into rows_by_key (list of seed-level scalars) + seen_seeds set
rows_by_key = defaultdict(list)
seen_seeds = defaultdict(set)
for (ds, method, horizon, seed), vals in raw.items():
    seen_seeds[(ds, method, horizon)].add(seed)
    for m in METRICS:
        rows_by_key[(ds, method, horizon, m)].append(vals[m])


def _agg(vals):
    if not vals:
        return None, None
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std


def _rank(values):
    """Return (best_idx_set, second_idx_set) over a list of (mean, std) where
    None entries are ignored. Lower is better."""
    valid = [(i, v[0]) for i, v in enumerate(values) if v[0] is not None]
    if not valid:
        return set(), set()
    valid.sort(key=lambda x: x[1])
    best = valid[0][1]
    best_set = {i for i, v in valid if abs(v - best) < 1e-9}
    rest = [v for v in valid if v[0] not in best_set]
    if not rest:
        return best_set, set()
    second = rest[0][1]
    second_set = {i for i, v in rest if abs(v - second) < 1e-9}
    return best_set, second_set


# ---------------------------------------------------------------------------
# Datasets actually populated (any cell in any group).
# ---------------------------------------------------------------------------
present_ds = set()
present_slug = set()
for (ds, slug, *_), vals in rows_by_key.items():
    if vals:
        present_ds.add(ds)
        present_slug.add(slug)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _fmt_tex(mean, std, is_best, is_second):
    if mean is None:
        return "--"
    s = f"{mean:.2f}$_{{\\pm{std:.2f}}}$"
    if is_best:
        return f"\\textbf{{{mean:.2f}$_{{\\pm{std:.2f}}}$}}"
    if is_second:
        return f"\\underline{{{mean:.2f}$_{{\\pm{std:.2f}}}$}}"
    return s


def build_tex(datasets, *, caption, label):
    # Group span widths in column order
    group_spans = []
    last = None
    for grp, _lbl, _slug in COLUMNS:
        if grp == last:
            group_spans[-1] = (grp, group_spans[-1][1] + 1)
        else:
            group_spans.append((grp, 1))
            last = grp
    n_cols = len(COLUMNS)
    # 3 leading meta cols (Dataset|Metric|Horizon) + n_cols method cols
    col_spec = "ccc|" + "|".join(["c" * w for _g, w in group_spans])

    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"\centering")
    out.append(r"\caption{" + caption + r"}")
    out.append(r"\label{" + label + r"}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{" + col_spec + r"}")
    out.append(r"\toprule")

    # Header row 1: group spans
    h1 = [
        r"\multirow{2}{*}{\textbf{Dataset}}",
        r"\multirow{2}{*}{\textbf{Metric}}",
        r"\multirow{2}{*}{\textbf{Horizon}}",
    ]
    for grp, w in group_spans:
        h1.append(r"\multicolumn{" + str(w) + r"}{c"
                  + ("|" if grp != group_spans[-1][0] else "")
                  + r"}{\textbf{" + grp + r"}}")
    out.append(" & ".join(h1) + r" \\")

    # cmidrules per group
    cmid = []
    col_cursor = 4  # first method col index (1-based) = 4
    for grp, w in group_spans:
        cmid.append(r"\cmidrule(lr){" + f"{col_cursor}-{col_cursor + w - 1}" + r"}")
        col_cursor += w
    out.append(" ".join(cmid))

    # Header row 2: column labels under the group spans
    h2 = ["", "", ""]
    for _grp, lbl, _slug in COLUMNS:
        h2.append(r"\textbf{" + lbl + r"}")
    out.append(" & ".join(h2) + r" \\")
    out.append(r"\midrule")

    n_rows_per_ds = len(METRICS) * len(HORIZONS)
    emitted = [d for d in datasets if d in present_ds]
    for di, ds in enumerate(emitted):
        for mi, met in enumerate(METRICS):
            for hi, h in enumerate(HORIZONS):
                # Compute (mean, std) for every column
                vals = []
                for _grp, _lbl, slug in COLUMNS:
                    raw = rows_by_key.get((ds, slug, h, met))
                    vals.append(_agg(raw))
                best_set, second_set = _rank(vals)

                cells = [_fmt_tex(m, s, i in best_set, i in second_set)
                         for i, (m, s) in enumerate(vals)]

                ds_cell = ""
                if mi == 0 and hi == 0:
                    ds_cell = r"\multirow{" + str(n_rows_per_ds) + r"}{*}{\textbf{" + ds + r"}}"
                met_cell = ""
                if hi == 0:
                    met_label = r"MAPE (\%)" if met == "MAPE" else met
                    met_cell = r"\multirow{" + str(len(HORIZONS)) + r"}{*}{" + met_label + r"}"

                out.append(" & ".join([ds_cell, met_cell, h] + cells) + r" \\")
            # cmidrule between metrics within a dataset
            if mi < len(METRICS) - 1:
                out.append(r"\cmidrule(lr){2-" + str(3 + n_cols) + r"}")
        if di < len(emitted) - 1:
            out.append(r"\midrule")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}%")
    out.append(r"}")
    out.append(r"\end{table*}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _fmt_md(mean, std, is_best, is_second):
    if mean is None:
        return "—"
    s = f"{mean:.2f} ± {std:.2f}"
    if is_best:
        return f"**{s}**"
    if is_second:
        return f"_{s}_"
    return s


def build_md(datasets):
    out = []
    out.append("# Newly-Added-Sensor Forecast Errors")
    out.append("")
    out.append("Mean ± std over seeds, computed on **newly added sensors only** "
               "(years where the graph did not grow are excluded). "
               "**Bold** = best per row, _underline_ = second best.")
    out.append("")
    out.append("Column groups and order follow `tables/main_table_full.tex`:")
    out.append("- **Static STGNN Backbones**: DCRNN, ASTGNN, TGCN, GWN")
    out.append("- **Naive Schemes**: Pretrain, Retrain, OL-NN, OL-AN")
    out.append("- **Evolving-Graph Continual**: TrafStm, PECPM, STKEC, EAC")
    out.append("- **Retrieval / TTC**: STRAP, ST-TTC")
    out.append("- **Other Static Backbones**: STID, STNorm, iTrans, DLinear")
    out.append("- **Ours**: A2TTA")
    out.append("")
    labels = [lbl for _g, lbl, _s in COLUMNS]
    header = "| Metric | Horizon | " + " | ".join(labels) + " |"
    align = "|:------:|:-------:|" + "|".join([":-:"] * len(labels)) + "|"

    for ds in datasets:
        if ds not in present_ds:
            continue
        out.append(f"## {ds}")
        out.append("")
        out.append(header)
        out.append(align)
        for met in METRICS:
            for hi, h in enumerate(HORIZONS):
                vals = []
                for _g, _lbl, slug in COLUMNS:
                    raw = rows_by_key.get((ds, slug, h, met))
                    vals.append(_agg(raw))
                best_set, second_set = _rank(vals)
                cells = [_fmt_md(m, s, i in best_set, i in second_set)
                         for i, (m, s) in enumerate(vals)]
                met_lbl = "**MAPE (%)**" if met == "MAPE" else f"**{met}**"
                met_cell = met_lbl if hi == 0 else ""
                out.append(f"| {met_cell} | {h} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------
def coverage_report():
    print("=== Coverage (seeds per (dataset, method) for new-sensor eval) ===")
    for ds in DATASET_PART1 + DATASET_PART2:
        if ds not in present_ds:
            print(f"  {ds:<7}  [NOT PRESENT]")
            continue
        line = []
        for _g, lbl, slug in COLUMNS:
            seeds = sorted(seen_seeds.get((ds, slug, "Avg"), []))
            mark = f"{len(seeds)}" if seeds else "·"
            line.append(f"{lbl}={mark}")
        print(f"  {ds:<7}  " + "  ".join(line))


# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
os.makedirs(TABLES_DIR, exist_ok=True)

caption_base = (r"Newly-added-sensor forecast errors, part {pi}/2 "
                r"({ds_list}; mean$\pm$std over seeds per dataset). "
                r"Years where the graph did not grow are excluded. "
                r"Baselines are grouped and ordered to match "
                r"\Cref{{tab:main_full}}. \textbf{{Bold}}: best, "
                r"\underline{{underline}}: second best.")

tex1_path = osp.join(TABLES_DIR, "tsas_new_sensors_part1.tex")
tex2_path = osp.join(TABLES_DIR, "tsas_new_sensors_part2.tex")
md_path = osp.join(TABLES_DIR, "tsas_new_sensors.md")

with open(tex1_path, "w") as f:
    f.write(build_tex(
        DATASET_PART1,
        caption=caption_base.format(pi=1, ds_list=", ".join(DATASET_PART1)),
        label="tab:tsas_new_sensors_part1",
    ))
print(f"[ok] wrote {tex1_path}")

with open(tex2_path, "w") as f:
    f.write(build_tex(
        DATASET_PART2,
        caption=caption_base.format(pi=2, ds_list=", ".join(DATASET_PART2)),
        label="tab:tsas_new_sensors_part2",
    ))
print(f"[ok] wrote {tex2_path}")

with open(md_path, "w") as f:
    f.write(build_md(DATASET_PART1 + DATASET_PART2))
print(f"[ok] wrote {md_path}")

print()
coverage_report()
