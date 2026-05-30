#!/bin/bash
# ===========================================================================
# PEMS-Stream 全流程重训脚本 (自动化版)
# ---------------------------------------------------------------------------
# 1. Retrain  : 每年从零训练 (2011~2017)，作为 Oracle 上界 & pretrain 的基座
# 2. AutoLink : 把 retrain 的 2011 权重软链到 pretrain_st_pems-<seed>/2011/
#               (因为 load_test_best_model 只按 logname 目录找，不读 --first_year_model_path)
# 3. Pretrain : 只用 2011 权重直接推理后续年份 (下界)
# 4~6.        : online-nn / online-an / TrafficStream 基线
# 7. PatchSTKEC: 自动把 conf/PEMS/stkec.json 里的 influence_path 占位替换成真实目录
# 8. STKEC / 9. EAC : 继续跑剩余方法
# ---------------------------------------------------------------------------
# 用法：
#   cd eac/
#   bash scripts/pems_run.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # 切到 eac/ 根目录

SEEDS_MAIN=(42 43 44 45 46)   # retrain / pretrain / online-nn / trafficstream
SEEDS_ALT=(51 52 53 54 55)    # online-an / eac
SEEDS_STKEC=(47 48 49 50 51)  # stkec

GPU_MAIN=2
GPU_STKEC=0
GPU_EAC=1

# ============================================================================
# 1. Retrain (Oracle 上界：每年都从零训练)
# ============================================================================
echo "==================== [1/8] Retrain ===================="
for seed in "${SEEDS_MAIN[@]}"; do
    python main.py --conf conf/PEMS/retrain_st_pems.json --gpuid $GPU_MAIN --seed "$seed"
done


# ============================================================================
# 2. AutoLink: 为 pretrain 建首年权重目录 (软链 retrain 产物)
# ============================================================================
echo "==================== [2/8] AutoLink retrain→pretrain ===================="
for seed in "${SEEDS_MAIN[@]}"; do
    src_dir="log/PEMS/retrain_st_pems-${seed}/2011"
    dst_dir="log/PEMS/pretrain_st_pems-${seed}/2011"
    if [ ! -d "$src_dir" ] || [ -z "$(ls -A "$src_dir"/*.pkl 2>/dev/null)" ]; then
        echo "  [skip] 找不到 $src_dir/*.pkl，请先跑 retrain"
        continue
    fi
    mkdir -p "$dst_dir"
    rm -f "$dst_dir"/*.pkl
    for f in "$src_dir"/*.pkl; do
        ln -sf "$(readlink -f "$f")" "$dst_dir/$(basename "$f")"
    done
    echo "  [ok] seed=$seed  $src_dir/*.pkl  ->  $dst_dir/"
done


# ============================================================================
# 3. Pretrain (下界：首年权重 + 后续年份不训练，直接测)
# ============================================================================
echo "==================== [3/8] Pretrain ===================="
for seed in "${SEEDS_MAIN[@]}"; do
    first_year_pkl=$(ls log/PEMS/retrain_st_pems-${seed}/2011/*.pkl 2>/dev/null | head -n 1)
    if [ -z "$first_year_pkl" ]; then
        echo "  [skip] seed=$seed 缺少 retrain 2011 权重"; continue
    fi
    python main.py --conf conf/PEMS/pretrain_st_pems.json \
        --load_first_year 1 \
        --first_year_model_path "$first_year_pkl" \
        --gpuid $GPU_MAIN --seed "$seed"
done


# ============================================================================
# 4. Online-NN (仅新节点增量微调)
# ============================================================================
echo "==================== [4/8] Online-NN ===================="
for seed in "${SEEDS_MAIN[@]}"; do
    python main.py --conf conf/PEMS/oneline_st_nn_pems.json --gpuid $GPU_MAIN --seed "$seed"
done


# ============================================================================
# 5. Online-AN (全节点 retrain 式微调)
# ============================================================================
echo "==================== [5/8] Online-AN ===================="
for seed in "${SEEDS_ALT[@]}"; do
    python main.py --conf conf/PEMS/oneline_st_an_pems.json --gpuid $GPU_MAIN --seed "$seed"
done


# ============================================================================
# 6. TrafficStream (EWC + Replay continual baseline)
# ============================================================================
echo "==================== [6/8] TrafficStream ===================="
for seed in "${SEEDS_MAIN[@]}"; do
    python main.py --conf conf/PEMS/trafficstream.json --gpuid $GPU_MAIN --seed "$seed"
done


# ============================================================================
# 7. PatchSTKEC: 自动修 influence_path 占位
# ============================================================================
echo "==================== [7/8] Patch STKEC influence_path ===================="
STKEC_INFLUENCE_DIR="log/PEMS/stkec_influence"
mkdir -p "$STKEC_INFLUENCE_DIR"
python - <<PY
import json, sys
from pathlib import Path
p = Path("conf/PEMS/stkec.json")
data = json.loads(p.read_text())
cur = data.get("influence_path", "")
target = "${STKEC_INFLUENCE_DIR}/"
if "Please enter" in cur or not cur:
    data["influence_path"] = target
    p.write_text(json.dumps(data, indent=4))
    print(f"  [ok] influence_path 已写入 -> {target}")
else:
    print(f"  [skip] influence_path 已是 {cur}")
PY

echo "==================== [7/8] STKEC ===================="
for seed in "${SEEDS_STKEC[@]}"; do
    python stkec_main.py --conf conf/PEMS/stkec.json --gpuid $GPU_STKEC --seed "$seed"
done


# ============================================================================
# 8. EAC (本论文主方法)
# ============================================================================
# 注意：conf/PEMS/eac.json 默认 "train": 0, "auto_test": 1 —— 即“只评测作者已上传权重”。
#       如果你要**从零训练 EAC**，请先把 JSON 改成 "train": 1, "auto_test": 0。
#       否则它会去加载 log/PEMS/eac-<seed>/2011/*.pkl，若不存在会报错。
echo "==================== [8/8] EAC ===================="
for seed in "${SEEDS_ALT[@]}"; do
    python main.py --conf conf/PEMS/eac.json --gpuid $GPU_EAC --seed "$seed"
done

echo "==================== ALL DONE ===================="
