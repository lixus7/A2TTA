#!/usr/bin/env python3
"""Summarise an HP sweep (SW_<DSID>__lr..__st..__wu..__s..csv) and emit, per
dataset, the BEST config (min Avg-MAE over seeds) in MAIN-TABLE format:
full MAE/RMSE/MAPE x {3,6,12,Avg} as 'mean ± std' (pop-std over seeds).

Per-seed value = mean over ALL years of csv[metric] at that horizon (same
convention as the main table / insert_stae_ours_col.py).

Usage: python scripts/hpsweep_best_maintable.py run_logs/<olanSW|staeFSW>_<ts>
"""
import csv, glob, os.path as osp, re, statistics as st, sys
from collections import defaultdict

TAG = re.compile(r"^SW_(P\d+|TFNSW)__lr([0-9.e+-]+)__st(\d+)__wu(\d+)__s(\d+)\.csv$")
HZ = ["3", "6", "12", "Avg"]
METS = ["MAE", "RMSE", "MAPE"]
BASE = ("1e-3", "3", "3")
DSID2NAME = {f"P{n}": (f"PEMS{n:02d}" if n != 0 else "?") for n in range(3, 13)}
DSID2NAME["TFNSW"] = "TFNSW"


def perseed(path):
    """{met: {hz: mean_over_years}} for one csv."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        h = r.get("horizon")
        if h in HZ:
            for m in METS:
                try:
                    acc[m][h].append(float(r[m]))
                except (ValueError, KeyError):
                    pass
    return {m: {h: st.mean(v) for h, v in d.items() if v} for m, d in acc.items()}


def main(root):
    # store[dsid][cfg][met][hz] = [per-seed means]
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for p in sorted(glob.glob(osp.join(root, "csv", "SW_*.csv"))):
        m = TAG.match(osp.basename(p))
        if not m:
            continue
        dsid, lr, stp, wu, _seed = m.groups()
        ph = perseed(p)
        for met, hz in ph.items():
            for h, v in hz.items():
                store[dsid][(lr, stp, wu)][met][h].append(v)
    if not store:
        print(f"[!] no SW_*.csv under {root}/csv"); return

    def cell(d, h):
        v = d.get(h)
        return f"{st.mean(v):.2f} ± {st.pstdev(v):.2f}" if v else "--"

    print(f"# HP-sweep best configs (main-table format) — {osp.basename(root)}\n")
    best_tbl = []
    for dsid in sorted(store, key=lambda x: (x != "TFNSW", x)):
        cfgs = store[dsid]
        ranked = sorted(
            cfgs.items(),
            key=lambda kv: st.mean(kv[1]["MAE"]["Avg"]) if kv[1].get("MAE", {}).get("Avg") else 9e9)
        bcfg, bd = ranked[0]
        base = cfgs.get(BASE)
        bavg = st.mean(bd["MAE"]["Avg"])
        bavg_base = st.mean(base["MAE"]["Avg"]) if base and base.get("MAE", {}).get("Avg") else None
        nseed = len(bd["MAE"]["Avg"])
        ds = DSID2NAME.get(dsid, dsid)
        delta = f"{bavg - bavg_base:+.2f} vs baseline" if bavg_base is not None else ""
        tag = "(==baseline)" if bcfg == BASE else f"(baseline Avg-MAE={bavg_base:.2f} {delta})" if bavg_base else ""
        print(f"## {ds}  best: lr={bcfg[0]} steps={bcfg[1]} warmup={bcfg[2]}  "
              f"[{nseed} seeds]  {tag}")
        print("| Metric | @3 | @6 | @12 | Avg |")
        print("|---|---|---|---|---|")
        for met in METS:
            d = bd.get(met, {})
            print(f"| {met} | {cell(d,'3')} | {cell(d,'6')} | {cell(d,'12')} | {cell(d,'Avg')} |")
        print()
        best_tbl.append((ds, bcfg, bavg, bavg_base))

    print("\n# Best-config Avg-MAE summary")
    print("| dataset | best lr/steps/wu | Avg MAE | baseline Avg MAE | Δ |")
    print("|---|---|---|---|---|")
    for ds, c, b, bb in best_tbl:
        d = f"{b-bb:+.2f}" if bb is not None else "—"
        bbs = f"{bb:.2f}" if bb is not None else "—"
        print(f"| {ds} | {c[0]}/{c[1]}/{c[2]} | {b:.2f} | {bbs} | {d} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: hpsweep_best_maintable.py run_logs/<root>"); sys.exit(1)
    main(sys.argv[1])
