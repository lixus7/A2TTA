"""
Summarise Ablation-D (knockout / leave-one-out) jobs.

Per-job CSVs tagged  KO_<bb>__<P#>__<method>__<arch>__s<seed>.csv
(bb in {an, stae}). For each (backbone, dataset) prints a markdown table:
  MAE@3 / @6 / @12 / Avg  (mean +/- std over seeds; years averaged per seed)
  + Delta Avg vs the Ours(full) row.

Usage: python scripts/knockoutD_summary.py run_logs/koD_<ts>
"""
from __future__ import annotations
import csv, glob, os.path as osp, sys
from collections import defaultdict
import numpy as np

TAG_RE = __import__("re").compile(
    r"^KO_(an|stae)__(P\d+)__([a-z_]+)__([a-z]+)__s(\d+)\.csv$")
HZ = ["3", "6", "12", "Avg"]

BB_LABEL = {"an": "Online-AN (TrafficStream)", "stae": "STAEformer"}

# (method, arch) -> (display label, order rank). Ours(full) is rank 0 = reference.
ROW = {
    ("tta_ctx_local", "film"):   ("Ours (full)", 0),
    ("tta_all", "film"):         ("- local clone", 1),
    ("tta_ctx_local", "affine"): ("- FiLM -> affine (naive)", 2),
    ("calibrator", "film"):      ("- online TTA (FiLM warm-up only)", 3),
    ("backbone", "film"):        ("- calibration stack -> backbone", 4),
}


def perhz_per_seed(paths):
    """{label: {hz: [seed_means]}} ; each seed mean = mean over years."""
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        m = TAG_RE.match(osp.basename(p))
        if not m:
            continue
        _bb, _ds, method, arch, _seed = m.groups()
        key = ROW.get((method, arch))
        if key is None:
            continue
        label = key[0]
        acc = defaultdict(list)
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("horizon") in HZ:
                    try:
                        acc[r["horizon"]].append(float(r["MAE"]))
                    except (ValueError, KeyError):
                        pass
        for h, v in acc.items():
            if v:
                out[label][h].append(float(np.mean(v)))
    return out


def main(root):
    groups = defaultdict(list)  # (bb, ds) -> [csv paths]
    for p in sorted(glob.glob(osp.join(root, "csv", "KO_*.csv"))):
        m = TAG_RE.match(osp.basename(p))
        if m:
            groups[(m.group(1), m.group(2))].append(p)
    if not groups:
        print(f"[!] no KO_*.csv under {osp.join(root, 'csv')}")
        return

    order = sorted(ROW.values(), key=lambda t: t[1])
    order_labels = [lbl for lbl, _ in order]

    for (bb, ds) in sorted(groups):
        store = perhz_per_seed(groups[(bb, ds)])
        if not store:
            continue
        ours_avg = (np.mean(store["Ours (full)"]["Avg"])
                    if store.get("Ours (full)", {}).get("Avg") else None)
        print(f"\n### {ds} - {BB_LABEL.get(bb, bb)}")
        print("| Knockout | MAE@3 | MAE@6 | MAE@12 | Avg | Δ Avg | n |")
        print("|---|---|---|---|---|---|---|")
        present = [l for l in order_labels if l in store] + \
                  [l for l in store if l not in order_labels]
        for lbl in present:
            hz = store[lbl]
            cells = []
            for h in HZ:
                v = hz.get(h)
                cells.append(f"{np.mean(v):.2f}±{np.std(v):.2f}" if v else "--")
            n = len(hz.get("Avg", []))
            avg = np.mean(hz["Avg"]) if hz.get("Avg") else None
            if lbl == "Ours (full)":
                d = "—"
            elif avg is not None and ours_avg is not None:
                d = f"{avg - ours_avg:+.2f}"
            else:
                d = "--"
            name = f"**{lbl}**" if lbl == "Ours (full)" else lbl
            print(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {d} | {n} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/knockoutD_summary.py run_logs/koD_<ts>")
        sys.exit(1)
    main(sys.argv[1])
