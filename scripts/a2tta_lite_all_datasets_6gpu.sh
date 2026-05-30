#!/bin/bash
# ===========================================================================
# A2TTA-Lite on all remaining PEMS datasets, 6-GPU parallel.
# ---------------------------------------------------------------------------
# Default GPU pool: 2 3 4 5 6 7  (skips GPU 0 and GPU 1)
# Default datasets: PEMS03 PEMS04 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12
#   (PEMS05 already done — its CSV remains untouched.)
# Methods (same matrix as PEMS05 run): backbone, calibrator, tta_random,
#   tta_recent, tta_error, a2tta_lite.
# ---------------------------------------------------------------------------
# Output:
#   run_logs/multi_a2tta_<ts>/csv/<DATASET>__<METHOD>__s<SEED>.csv
#                         /logs/<DATASET>__<METHOD>__s<SEED>.log
#                         /results.csv         (concat of per-job CSVs)
#                         /dispatcher.log
# ---------------------------------------------------------------------------
# Usage:
#   cd eac/
#   NOHUP=1 bash scripts/a2tta_lite_all_datasets_6gpu.sh
#   tail -f run_logs/multi_a2tta_*/dispatcher.log
#
# Tweak:
#   GPUS="2 3 4 5"            DATASETS="PEMS03 PEMS04"   \
#   SEEDS="51 52"             METHODS="a2tta_lite tta_random" \
#   NOHUP=1 bash scripts/a2tta_lite_all_datasets_6gpu.sh
#
#   FAST_DEV_RUN=1            (1 year × 4 batches per job — sanity check)
#   RESUME_FROM=<sweep_root>  (skip jobs whose CSV already exists)
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
GPUS=${GPUS:-"2 3 4 5 6 7"}
# PEMS05 is intentionally included so its results are regenerated with the
# tuned hyperparameters from sweep_a2tta_pems05_20260506_183335 (lr=1e-3,
# steps=3, budget=0.25, pool=512 — see scripts/a2tta_sweep_summarize.py).
DATASETS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
SEEDS=${SEEDS:-"51 52 53 54 55"}
METHODS=${METHODS:-"backbone calibrator tta_random tta_recent tta_error a2tta_lite"}
BACKBONE_LOGNAME=${BACKBONE_LOGNAME:-"oneline_st_an"}     # template; <DATASET-lower> appended
BACKBONE_FALLBACK=${BACKBONE_FALLBACK:-"retrain_st"}      # template
WARMUP_EPOCHS=${WARMUP_EPOCHS:-3}
EVAL_BATCH=${EVAL_BATCH:-64}
# --- tuned defaults (from sweep top-1 a2tta_lite config; see Δ −0.24 vs Online-AN) ---
ADAPT_LR=${ADAPT_LR:-1e-3}
ADAPT_STEPS=${ADAPT_STEPS:-3}
ADAPT_EVERY=${ADAPT_EVERY:-1}
BUDGET_FRAC=${BUDGET_FRAC:-0.25}
POOL_SIZE=${POOL_SIZE:-512}
LAMBDA_CONS=${LAMBDA_CONS:-0.05}
LAMBDA_REG=${LAMBDA_REG:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NODE_EMB_DIM=${NODE_EMB_DIM:-16}
FAST_DEV_RUN=${FAST_DEV_RUN:-0}
RESUME_FROM=${RESUME_FROM:-""}
DRY_RUN=${DRY_RUN:-0}

if [[ -n "$RESUME_FROM" ]]; then
    SWEEP_ROOT=$RESUME_FROM
    [[ -d "$SWEEP_ROOT" ]] || { echo "[err] RESUME_FROM=$SWEEP_ROOT not found"; exit 1; }
    echo "[multi] RESUMING under $SWEEP_ROOT"
else
    TS=$(date +%Y%m%d_%H%M%S)
    SWEEP_ROOT=run_logs/multi_a2tta_$TS
fi
mkdir -p "$SWEEP_ROOT/csv" "$SWEEP_ROOT/logs"

# ---------------------------------------------------------------------------
# nohup self-bg
# ---------------------------------------------------------------------------
if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    EAC_BG=1 nohup bash "$0" "$@" > "$SWEEP_ROOT/dispatcher.log" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $SWEEP_ROOT/dispatcher.log"
    echo "[nohup] sweep root: $SWEEP_ROOT"
    exit 0
fi

echo "[multi] root=$SWEEP_ROOT"
echo "[multi] GPUs='$GPUS'  datasets='$DATASETS'  seeds='$SEEDS'  methods='$METHODS'"

# ---------------------------------------------------------------------------
# GPU semaphore (FIFO) — pool seeded from explicit GPU list (NOT 0..N-1)
# ---------------------------------------------------------------------------
FIFO=$SWEEP_ROOT/.gpu_fifo
[[ -p "$FIFO" ]] || mkfifo "$FIFO"
exec 3<>"$FIFO"
for g in $GPUS; do echo "$g" >&3; done

JOBS_ENQ=0
JOBS_SKIP=0

run_job() {
    local tag=$1; shift
    local csv="$SWEEP_ROOT/csv/$tag.csv"
    local log="$SWEEP_ROOT/logs/$tag.log"
    JOBS_ENQ=$((JOBS_ENQ+1))

    if [[ -f "$csv" ]]; then
        JOBS_SKIP=$((JOBS_SKIP+1))
        echo "[$JOBS_ENQ skip] $tag (csv exists)"
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$JOBS_ENQ dry] $tag $*"
        return
    fi

    # Acquire GPU (BLOCKS until one frees up)
    local gpu
    read -u 3 gpu

    {
        local start_ts=$(date +%s)
        echo "[gpu=$gpu start $(date +%H:%M:%S)] $tag" >> "$SWEEP_ROOT/dispatcher.log"
        python a2tta_main.py "$@" \
            --gpuid "$gpu" \
            --csv_path "$csv" \
            --eval_batch_size "$EVAL_BATCH" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            --fast_dev_run "$FAST_DEV_RUN" \
            >> "$log" 2>&1
        local rc=$?
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        echo "[gpu=$gpu done  $(date +%H:%M:%S) rc=$rc t=${elapsed}s] $tag" >> "$SWEEP_ROOT/dispatcher.log"
        if [[ "$rc" != "0" ]]; then
            echo "[$JOBS_ENQ FAIL rc=$rc] $tag — see $log" >> "$SWEEP_ROOT/dispatcher.log"
            rm -f "$csv"
        fi
        echo "$gpu" >&3
    } &
}

# ---------------------------------------------------------------------------
# Enqueue: dataset × method × seed
# ---------------------------------------------------------------------------
for DATASET in $DATASETS; do
    DD=${DATASET#PEMS}                 # e.g. PEMS03 → 03
    LOWER=$(echo "$DATASET" | tr 'A-Z' 'a-z')   # e.g. pems03
    CONF="conf/${DATASET}/a2tta_lite_${LOWER}.json"
    BB_LOG="${BACKBONE_LOGNAME}_${LOWER}"        # e.g. oneline_st_an_pems03
    BB_FB="${BACKBONE_FALLBACK}_${LOWER}"        # e.g. retrain_st_pems03

    if [[ ! -f "$CONF" ]]; then
        echo "[skip dataset $DATASET] $CONF not found"
        continue
    fi

    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            TAG="${DATASET}__${METHOD}__s${SEED}"
            run_job "$TAG" \
                --conf "$CONF" \
                --logname "a2tta_${METHOD}_${LOWER}" \
                --method "$METHOD" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --backbone_ckpt_logname "$BB_LOG" \
                --backbone_ckpt_logname_fallback "$BB_FB" \
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
                --lambda_reg "$LAMBDA_REG"
        done
    done
done

echo "[multi] enqueued=$JOBS_ENQ  skipped=$JOBS_SKIP"
echo "[multi] waiting for outstanding jobs..."
wait
exec 3>&-
rm -f "$FIFO"

# ---------------------------------------------------------------------------
# Concatenate per-job CSVs
# ---------------------------------------------------------------------------
RESULTS="$SWEEP_ROOT/results.csv"
shopt -s nullglob
ALL_CSV=( "$SWEEP_ROOT/csv/"*.csv )
shopt -u nullglob
if [[ ${#ALL_CSV[@]} -gt 0 ]]; then
    head -n 1 "${ALL_CSV[0]}" > "$RESULTS"
    for f in "${ALL_CSV[@]}"; do
        tail -n +2 "$f" >> "$RESULTS"
    done
    echo "[multi] combined CSV → $RESULTS ($(wc -l < "$RESULTS") lines, ${#ALL_CSV[@]} jobs)"
fi

# ---------------------------------------------------------------------------
# Per-dataset compact summary
# ---------------------------------------------------------------------------
if [[ -f "$RESULTS" ]]; then
    for DATASET in $DATASETS; do
        echo
        echo "=========================================================="
        echo "  Summary: $DATASET"
        echo "=========================================================="
        python scripts/a2tta_summarize.py "$RESULTS" --dataset "$DATASET" \
            --ref_avg_mae 11.10 || true
    done
fi

echo "==================== A2TTA-Lite MULTI-DATASET DONE ===================="
echo "Sweep root  : $SWEEP_ROOT"
echo "Results CSV : $RESULTS"
echo "Per-job logs: $SWEEP_ROOT/logs/"
