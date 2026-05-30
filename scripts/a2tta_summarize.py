"""
Read the A2TTA-Lite CSV and print a compact comparison table per dataset.

Usage:
  python scripts/a2tta_summarize.py run_logs/a2tta_lite_pems05_results.csv
  python scripts/a2tta_summarize.py run_logs/a2tta_lite_pems05_results.csv --dataset PEMS05

The CSV is one row per (dataset, method, seed, year, horizon). We average over
years, then mean±std across seeds. Output mirrors the table in the prompt:

  Method          MAE@3   MAE@6   MAE@12   Avg MAE
  backbone        10.40   11.01   12.25    11.11
  calibrator      ...
  ...

A2TTA-Lite Avg MAE row is highlighted with a >> marker if it beats the
configured Online-AN reference (default 11.10 on PEMS05).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def read_rows(path: str):
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def summarize(path: str, dataset: str | None = None, ref_avg_mae: float = 11.10):
    # rows[method][horizon] = list of (seed, year, MAE)
    rows = defaultdict(lambda: defaultdict(list))
    for r in read_rows(path):
        if dataset and r["dataset"] != dataset:
            continue
        try:
            mae = float(r["MAE"])
        except (ValueError, KeyError):
            continue
        rows[r["method"]][r["horizon"]].append((r["seed"], r["year"], mae))

    if not rows:
        print(f"[!] no rows for dataset={dataset!r} in {path}")
        return

    # Average over years per (method, seed, horizon), then mean/std across seeds.
    methods_order = ["backbone", "calibrator",
                     "tta_random", "tta_recent", "tta_error",
                     "a2tta_lite", "tta_all"]
    methods_present = [m for m in methods_order if m in rows] + \
                      [m for m in rows if m not in methods_order]

    horizons = ["3", "6", "12", "Avg"]
    print(f"\n=== {dataset or '(all datasets)'}  (mean±std over seeds, MAE) ===\n")
    header = f"{'Method':<14}" + "".join(f"{'MAE@'+h:>14}" if h != 'Avg' else f"{'Avg MAE':>14}" for h in horizons)
    print(header)
    print("-" * len(header))

    rows_to_print = []
    for m in methods_present:
        cols = []
        avg_mae_mean = None
        for h in horizons:
            seed_means = defaultdict(list)
            for s, _y, mae in rows[m].get(h, []):
                seed_means[s].append(mae)
            seed_avgs = [sum(v) / len(v) for v in seed_means.values() if v]
            if not seed_avgs:
                cols.append("    -")
                continue
            mean = sum(seed_avgs) / len(seed_avgs)
            if len(seed_avgs) > 1:
                var = sum((x - mean) ** 2 for x in seed_avgs) / (len(seed_avgs) - 1)
                std = var ** 0.5
                cols.append(f"{mean:>7.2f}±{std:<5.2f}")
            else:
                cols.append(f"{mean:>9.2f}")
            if h == "Avg":
                avg_mae_mean = mean
        rows_to_print.append((m, cols, avg_mae_mean))

    for m, cols, avg in rows_to_print:
        marker = ""
        if m == "a2tta_lite" and avg is not None and avg < ref_avg_mae:
            marker = f"  >> beats Online-AN ({ref_avg_mae:.2f})"
        elif m == "a2tta_lite" and avg is not None:
            marker = f"  (Online-AN ref {ref_avg_mae:.2f})"
        line = f"{m:<14}" + "".join(f"{c:>14}" for c in cols) + marker
        print(line)
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    p.add_argument("--dataset", type=str, default="PEMS05")
    p.add_argument("--ref_avg_mae", type=float, default=11.10,
                   help="Reference (Online-AN Avg MAE) to compare A2TTA-Lite against.")
    args = p.parse_args()
    summarize(args.csv, args.dataset, args.ref_avg_mae)
