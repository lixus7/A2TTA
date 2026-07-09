#!/bin/bash
# Quick interactive (non-PBS) runner for the TimesFM zero-shot baseline.
# Run from the eac/ root after scripts/setup_timesfm_env.sh has been done.
#
#   bash scripts/timesfm_run.sh                       # PEMS03, GPU 0, all years
#   DATASETS="PEMS03 PEMS04" GPU=1 bash scripts/timesfm_run.sh
#   FAST_DEV_RUN=2 bash scripts/timesfm_run.sh        # smoke test: first 2 years
set -euo pipefail

ENV_NAME=${ENV_NAME:-timesfm}
GPU=${GPU:-0}
SEED=${SEED:-51}
DATASETS=${DATASETS:-"PEMS03"}
FAST_DEV_RUN=${FAST_DEV_RUN:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
export HF_HOME=${HF_HOME:-./hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

source $CONDA
conda activate "$ENV_NAME"
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
OUT=run_logs/timesfm_local_$TS
mkdir -p "$OUT"
CSV="$OUT/results.csv"

for ds in $DATASETS; do
    lower=$(echo "$ds" | tr 'A-Z' 'a-z')
    echo "==================== TimesFM $ds (gpu=$GPU seed=$SEED) ===================="
    python timesfm_main.py \
        --conf "conf/${ds}/timesfm_${lower}.json" \
        --dataset "$ds" \
        --seed "$SEED" \
        --gpuid "$GPU" \
        --logname "timesfm_${lower}" \
        --csv_path "$CSV" \
        --chunk_size "$CHUNK_SIZE" \
        --fast_dev_run "$FAST_DEV_RUN" \
        --max_samples "$MAX_SAMPLES" \
        2>&1 | tee "$OUT/${ds}.log"
done
echo "results -> $CSV"
