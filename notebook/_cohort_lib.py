"""Shared data layer: build per-station first-seen year (cohort) from PEMS meta snapshots.
Import this from render scripts:  from _cohort_lib import build_cohort, all_cohorts, DISTRICTS, year_range
"""
import os, re, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_DIR = os.path.join(ROOT, "meta")
DISTRICTS = ["03", "04", "05", "06", "07", "08", "10", "11", "12"]


def meta_files(d):
    out = []
    for f in glob.glob(os.path.join(META_DIR, f"d{d}_text_meta_*.txt")):
        m = re.search(r"_(\d{4})_\d{2}_\d{2}\.txt$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def build_cohort(d):
    """Return (DataFrame[ID,Latitude,Longitude,Fwy,cohort], snapshot_years).
    Scan snapshots chronologically; each station ID keeps the first year it appears."""
    files, seen, parts = meta_files(d), set(), []
    for yr, f in files:
        df = pd.read_csv(f, sep="\t", usecols=["ID", "Latitude", "Longitude", "Fwy"]) \
               .dropna(subset=["Latitude", "Longitude"])
        new = df[~df["ID"].isin(seen)].copy()
        new["cohort"] = yr
        parts.append(new)
        seen.update(new["ID"].tolist())
    out = pd.concat(parts, ignore_index=True) if parts else \
        pd.DataFrame(columns=["ID", "Latitude", "Longitude", "Fwy", "cohort"])
    return out, [y for y, _ in files]


def all_cohorts():
    cohorts, snap_years = {}, {}
    for d in DISTRICTS:
        cohorts[d], snap_years[d] = build_cohort(d)
    return cohorts, snap_years


def year_range():
    cohorts, _ = all_cohorts()
    ys = np.concatenate([cohorts[d]["cohort"].values for d in DISTRICTS if len(cohorts[d])])
    return int(ys.min()), int(ys.max())
