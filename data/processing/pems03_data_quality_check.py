"""PEMS03 每年数据质量检查: 0 值、缺失值、异常值统计 (基于 RawData/*.npz)"""
import os
import numpy as np
import pandas as pd

DISTRICT = "pems03"
RAW_DIR = f"./preprocessed/{DISTRICT}/RawData"
YEARS = list(range(2001, 2026))

# 高值阈值: 5min 流量超过该值视为异常
HIGH_THRESHOLD = 1000

rows = []
header = (f"{'Year':>4} | {'T':>6} {'N':>4} | {'Zero%':>7} {'AllZeroT%':>10} {'DeadSens':>8} "
          f"| {'NaN%':>6} {'Neg%':>6} {'High%':>6} "
          f"| {'Min':>6} {'Max':>8} {'Mean':>7} {'Median':>7} {'Std':>7}")
print(header)
print("-" * len(header))

for year in YEARS:
    npz_path = os.path.join(RAW_DIR, f"{year}.npz")
    if not os.path.exists(npz_path):
        print(f"[SKIP] {year}")
        continue

    data = np.load(npz_path)["x"]  # (T, N) float32
    T, N = data.shape
    total = T * N

    nan_mask = np.isnan(data)
    nan_count = int(nan_mask.sum())
    filled = np.where(nan_mask, 0.0, data)

    zero_mask = (filled == 0) & (~nan_mask)
    zero_count = int(zero_mask.sum())

    neg_count = int((filled < 0).sum())
    high_count = int((filled > HIGH_THRESHOLD).sum())

    # 整个时刻所有传感器都为 0 的行数 (288 步 = 1 天)
    all_zero_rows = int(((filled == 0).all(axis=1)).sum())

    # "死亡"传感器: 整年都为 0 的传感器数
    dead_sensors = int(((filled == 0).all(axis=0)).sum())

    valid = data[~nan_mask]
    vmin = float(valid.min()) if valid.size else float("nan")
    vmax = float(valid.max()) if valid.size else float("nan")
    vmean = float(valid.mean()) if valid.size else float("nan")
    vmed = float(np.median(valid)) if valid.size else float("nan")
    vstd = float(valid.std()) if valid.size else float("nan")

    rows.append({
        "year": year, "T": T, "N": N,
        "zero_pct": zero_count / total * 100,
        "all_zero_t_pct": all_zero_rows / T * 100,
        "all_zero_t_count": all_zero_rows,
        "dead_sensors": dead_sensors,
        "nan_pct": nan_count / total * 100,
        "neg_pct": neg_count / total * 100,
        "high_pct": high_count / total * 100,
        "min": vmin, "max": vmax, "mean": vmean, "median": vmed, "std": vstd,
    })

    print(f"{year:>4} | {T:>6} {N:>4} | "
          f"{zero_count/total*100:>6.2f}% {all_zero_rows/T*100:>9.2f}% {dead_sensors:>8} "
          f"| {nan_count/total*100:>5.2f}% {neg_count/total*100:>5.2f}% {high_count/total*100:>5.2f}% "
          f"| {vmin:>6.1f} {vmax:>8.1f} {vmean:>7.2f} {vmed:>7.2f} {vstd:>7.2f}")

    del data, filled, nan_mask, zero_mask, valid

out_df = pd.DataFrame(rows)
out_path = os.path.join(".", f"{DISTRICT}_data_quality.csv")
out_df.to_csv(out_path, index=False)
print(f"\n[Saved] {out_path}")

# 额外: 按传感器层面统计 "几乎全 0" 的传感器分布
print("\n=== Per-sensor zero-ratio distribution (per year) ===")
print(f"{'Year':>4} | {'p50':>6} {'p90':>6} {'p99':>6} | {'>=50% zero':>11} {'>=90% zero':>11} {'100% zero':>11}")
print("-" * 80)
for year in YEARS:
    npz_path = os.path.join(RAW_DIR, f"{year}.npz")
    if not os.path.exists(npz_path):
        continue
    data = np.load(npz_path)["x"]
    filled = np.nan_to_num(data, nan=0.0)
    per_sensor_zero = (filled == 0).mean(axis=0)  # 每个传感器的 0 值占比
    p50 = np.percentile(per_sensor_zero, 50)
    p90 = np.percentile(per_sensor_zero, 90)
    p99 = np.percentile(per_sensor_zero, 99)
    n_half = int((per_sensor_zero >= 0.5).sum())
    n_90 = int((per_sensor_zero >= 0.9).sum())
    n_full = int((per_sensor_zero >= 1.0 - 1e-9).sum())
    print(f"{year:>4} | {p50*100:>5.1f}% {p90*100:>5.1f}% {p99*100:>5.1f}% "
          f"| {n_half:>11} {n_90:>11} {n_full:>11}")
    del data, filled, per_sensor_zero
