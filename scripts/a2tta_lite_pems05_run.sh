#!/bin/bash
# ===========================================================================
# A2TTA-Lite (Active Adaptive Test-Time Adaptation) on PEMS05.
# ---------------------------------------------------------------------------
# Backbone : Online-AN per-year .pkl checkpoints (oneline_st_an_pems05-<seed>/<year>/*.pkl)
#            Falls back to retrain_st_pems05 if missing.
# Method   : freeze backbone + add small residual calibrator (zero-init).
#            Optional warm-up on each year's train+val split (backbone frozen).
#            Online causal eval: delayed-label candidate pool, active selection,
#            adapt only the calibrator at test time.
# ---------------------------------------------------------------------------
# Output   : log/PEMS05/a2tta_*-<seed>/*.log  (per-year metrics)
#            run_logs/a2tta_lite_results.csv  (per year × method × seed × horizon)
# Usage    :
#   cd eac/
#   bash scripts/a2tta_lite_pems05_run.sh                    # full matrix, default GPU
#   GPU=2 SEEDS="51" bash scripts/a2tta_lite_pems05_run.sh   # single seed on GPU 2
#   FAST_DEV_RUN=1 bash scripts/a2tta_lite_pems05_run.sh     # debug: 1 year, 4 batches
#   METHODS="a2tta_lite" bash scripts/a2tta_lite_pems05_run.sh  # only main method
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."  # → eac/

# ---------------------------------------------------------------------------
# Background mode: NOHUP=1 bash scripts/a2tta_lite_pems05_run.sh
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    LOG_FILE="run_logs/a2tta_pems05_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    echo "[nohup] PID=$!"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Knobs (env-overridable)
# ---------------------------------------------------------------------------
GPU=${GPU:-1}
# Online-AN seeds 51..55 are present in this repo; default to those so the
# backbone .pkl always resolves. Override with SEEDS="42 43" etc.
SEEDS=${SEEDS:-"51 52 53 54 55"}
BACKBONE_LOGNAME=${BACKBONE_LOGNAME:-"oneline_st_an_pems05"}
BACKBONE_FALLBACK=${BACKBONE_FALLBACK:-"retrain_st_pems05"}
# Methods to run (space-separated). The full ablation matrix is:
#   backbone     : frozen base, no calibrator
#   calibrator   : warmed-up calibrator, no online TTA  (variant B)
#   tta_random   : delayed-label TTA, random selection  (variant C)
#   tta_recent   : delayed-label TTA, most recent       (variant D)
#   tta_error    : delayed-label TTA, error-only score  (variant E)
#   a2tta_lite   : full active selection (err+unc+shift+recency)  (variant F)
#   tta_all      : delayed-label TTA, full pool (upper bound)     (optional G)
METHODS=${METHODS:-"backbone calibrator tta_random tta_recent tta_error a2tta_lite"}

# Hyperparameters
ADAPT_LR=${ADAPT_LR:-3e-4}
ADAPT_STEPS=${ADAPT_STEPS:-1}
ADAPT_EVERY=${ADAPT_EVERY:-1}
BUDGET_FRAC=${BUDGET_FRAC:-0.25}
POOL_SIZE=${POOL_SIZE:-512}
LAMBDA_CONS=${LAMBDA_CONS:-0.05}
LAMBDA_REG=${LAMBDA_REG:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NODE_EMB_DIM=${NODE_EMB_DIM:-16}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
WARMUP_LR=${WARMUP_LR:-1e-3}
EVAL_BATCH=${EVAL_BATCH:-64}
FAST_DEV_RUN=${FAST_DEV_RUN:-0}
CSV_PATH=${CSV_PATH:-"run_logs/a2tta_lite_pems05_results.csv"}

echo "[A2TTA] GPU=$GPU SEEDS='$SEEDS' METHODS='$METHODS'"
echo "[A2TTA] backbone=$BACKBONE_LOGNAME (fallback=$BACKBONE_FALLBACK)"
echo "[A2TTA] adapt_lr=$ADAPT_LR steps=$ADAPT_STEPS budget=$BUDGET_FRAC pool=$POOL_SIZE"
echo "[A2TTA] λ_cons=$LAMBDA_CONS hidden=$HIDDEN_DIM warmup=$WARMUP_EPOCHS"
echo "[A2TTA] csv=$CSV_PATH fast_dev_run=$FAST_DEV_RUN"

run_one() {
    local METHOD=$1
    local SEED=$2
    local LOGNAME="a2tta_${METHOD}_pems05"
    echo "==================== [A2TTA] method=$METHOD seed=$SEED ===================="
    python a2tta_main.py \
        --conf conf/PEMS05/a2tta_lite_pems05.json \
        --logname "$LOGNAME" \
        --method "$METHOD" \
        --dataset PEMS05 \
        --gpuid "$GPU" \
        --seed "$SEED" \
        --backbone_ckpt_logname "$BACKBONE_LOGNAME" \
        --backbone_ckpt_logname_fallback "$BACKBONE_FALLBACK" \
        --backbone_method TrafficStream \
        --freeze_backbone 1 \
        --adapter_hidden_dim "$HIDDEN_DIM" \
        --node_emb_dim "$NODE_EMB_DIM" \
        --adapt_lr "$ADAPT_LR" \
        --adapt_steps "$ADAPT_STEPS" \
        --adapt_every_batches "$ADAPT_EVERY" \
        --budget_frac "$BUDGET_FRAC" \
        --candidate_pool_size "$POOL_SIZE" \
        --lambda_cons "$LAMBDA_CONS" \
        --lambda_reg "$LAMBDA_REG" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --warmup_lr "$WARMUP_LR" \
        --eval_batch_size "$EVAL_BATCH" \
        --csv_path "$CSV_PATH" \
        --fast_dev_run "$FAST_DEV_RUN"
}

for SEED in $SEEDS; do
    for METHOD in $METHODS; do
        run_one "$METHOD" "$SEED"
    done
done

echo "==================== A2TTA-Lite ALL DONE ===================="
echo "[A2TTA] Results CSV: $CSV_PATH"
