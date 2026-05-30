#!/bin/bash
# ===========================================================================
# ST-TTC baseline runner — covers PEMS03/04/05/06/07/08/10/11/12 (5 seeds each).
# ---------------------------------------------------------------------------
# Method  : retrain-style STGNN backbone trained per-year, paired with an
#           online spectral-domain calibrator (FRPlusModule) applied at test.
#           Port of https://github.com/Onedean/ST-TTC (arxiv 2506.00635).
# Configs : conf/PEMS<NN>/sttc_pems<NN>.json
# Per-year metric logs land at: log/<DATASET>/sttc_pems<NN>-<seed>/sttc_pems<NN>.log
# ---------------------------------------------------------------------------
# 用法：
#   cd eac/
#   bash scripts/sttc_run.sh                                   # 顺序跑默认全部
#   NOHUP=1 bash scripts/sttc_run.sh                           # 后台跑, 日志到 run_logs/sttc_*.log
#   GPU=0 bash scripts/sttc_run.sh                             # 全部用 GPU 0
#   DATASETS="pems05 pems06 pems10" bash scripts/sttc_run.sh   # 只跑指定数据集
#   SEEDS="42 43" bash scripts/sttc_run.sh                     # 自定义 seed 列表
#
# 续跑剩余数据集 (PEMS05/06/10 已完成，只跑剩下 6 个):
#   DATASETS="pems03 pems04 pems07 pems08 pems11 pems12" \
#       NOHUP=1 bash scripts/sttc_run.sh
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # 切到 eac/ 根目录

# ---------------------------------------------------------------------------
# 后台模式: NOHUP=1 触发, 日志落到 eac/run_logs/sttc_<时间戳>.log
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/sttc_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

# Match SEEDS_MAIN from per-dataset run scripts so ST-TTC is comparable to
# Retrain / Pretrain / Online-NN / TrafficStream columns in tables/main_table.tex.
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46}"
read -r -a DATASETS <<< "${DATASETS:-pems03 pems04 pems05 pems06 pems07 pems08 pems10 pems11 pems12}"

GPU=${GPU:-0}
echo "[GPU]      GPU=$GPU"
echo "[SEEDS]    ${SEEDS[*]}"
echo "[DATASETS] ${DATASETS[*]}"

run_one() {
    local ds="$1"        # e.g. pems05
    local DS_UPPER       # e.g. PEMS05
    DS_UPPER=$(echo "$ds" | tr '[:lower:]' '[:upper:]')
    local conf="conf/${DS_UPPER}/sttc_${ds}.json"

    if [ ! -f "$conf" ]; then
        echo "  [skip] missing config: $conf"
        return
    fi

    echo "==================== ST-TTC :: ${DS_UPPER} ===================="
    for seed in "${SEEDS[@]}"; do
        echo "-------- seed=${seed} (${DS_UPPER}) --------"
        python main.py --conf "$conf" --gpuid "$GPU" --seed "$seed"
    done
}

for ds in "${DATASETS[@]}"; do
    run_one "$ds"
done

echo "==================== ST-TTC ALL DONE ===================="
echo "[hint] per-year metrics:    log/<DATASET>/sttc_<dataset>-<seed>/sttc_<dataset>.log"
echo "[hint] aggregate stdout log: run_logs/sttc_*.log (if NOHUP=1 was used)"
