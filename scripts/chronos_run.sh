#!/bin/bash
# Local / single-GPU Chronos-2 zero-shot baseline runner (no PBS).
# For interactive GPU nodes or quick debugging. For the full 4xH200 sweep use
# scripts/katana_chronos_4gpu.sh instead.
#
# Usage:
#   bash scripts/chronos_run.sh                       # all datasets, GPU 0
#   GPU=1 DATASETS="pems03 pems04" bash scripts/chronos_run.sh
#   YEARS="2001 2002" DATASETS=pems03 bash scripts/chronos_run.sh   # subset of years
#   NOHUP=1 bash scripts/chronos_run.sh               # detach + log to run_logs/
set -euo pipefail
cd "$(dirname "$0")/.."

source ~/.bashrc 2>/dev/null || true
source $CONDA 2>/dev/null || true
conda activate chronos

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

GPU=${GPU:-0}
DATASETS=${DATASETS:-"pems03 pems04 pems05 pems06 pems07 pems08 pems10 pems11 pems12"}
MODEL=${MODEL:-"amazon/chronos-2"}
MODE=${MODE:-multivariate}      # multivariate | univariate
GROUPING=${GROUPING:-graph}     # graph | index (multivariate only)
GROUP_SIZE=${GROUP_SIZE:-64}
BATCH_SIZE=${BATCH_SIZE:-512}
DTYPE=${DTYPE:-bfloat16}
SEED=${SEED:-42}
YEARS=${YEARS:-""}   # empty = all years discovered by the python script

if [[ -n "${NOHUP:-}" ]]; then
    TS=$(date +%Y%m%d_%H%M%S)
    mkdir -p run_logs
    LOG="run_logs/chronos_g${GPU}_${TS}.log"
    echo "[chronos] logging to $LOG"
    exec > >(tee -a "$LOG") 2>&1
fi

for DS in $DATASETS; do
    COMMON=(--model "$MODEL" --gpuid "$GPU" --mode "$MODE" --grouping "$GROUPING"
            --group-size "$GROUP_SIZE" --batch-size "$BATCH_SIZE" --dtype "$DTYPE" --seed "$SEED")
    if [[ -n "$YEARS" ]]; then
        for Y in $YEARS; do
            echo "[chronos] === $DS $Y (gpu $GPU, $MODE) ==="
            python chronos_baseline.py --dataset "$DS" --year "$Y" "${COMMON[@]}"
        done
    else
        echo "[chronos] === $DS all years (gpu $GPU, $MODE) ==="
        python chronos_baseline.py --dataset "$DS" "${COMMON[@]}"
    fi
done
echo "[chronos] sweep done. metrics -> run_logs/chronos/<dataset>/<dataset>_results.csv"
