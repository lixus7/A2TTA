#!/bin/bash
# ===========================================================================
# Extra baselines from the STBP paper (ICLR'26) + DLinear, on all 9 PEMS
# datasets (5 seeds each).
# ---------------------------------------------------------------------------
# Methods covered (6 lightweight re-impls, see src/model/model.py docstrings):
#   GWN          (Graph WaveNet, Wu et al. 2019)
#   STID         (Shao et al. 2022, node-MLP variant)
#   ITRANSFORMER (Liu et al. 2024, N-as-tokens)
#   DLINEAR      (Zeng et al. 2023, trend + seasonal linear)
#   STNORM       (Deng et al. KDD 2021, SNorm+TNorm on Wavenet TCN)
#   STAEFORMER   (Liu et al. CIKM 2023, ST adaptive emb + temporal/spatial attn)
#
# Configs:   conf/<DATASET>/retrain_<method>_<dataset>.json
# Per-year metric logs: log/<DATASET>/retrain_<method>_<dataset>-<seed>/...
# Aggregate stdout log: run_logs/extra_baselines_<timestamp>.log  (NOHUP=1)
# ---------------------------------------------------------------------------
# Usage:
#   cd eac/
#   bash scripts/extra_baselines_run.sh                              # all 9 × 6 × 5
#   NOHUP=1 bash scripts/extra_baselines_run.sh                      # background + log
#   GPU=0 bash scripts/extra_baselines_run.sh                        # pin GPU
#   DATASETS="PEMS03 PEMS04" bash scripts/extra_baselines_run.sh     # subset of datasets
#   METHODS="gwn stid" bash scripts/extra_baselines_run.sh           # subset of methods
#   SEEDS="42 43 44 45 46" bash scripts/extra_baselines_run.sh       # override seeds
#
# Seed pool defaults to 42-46 to align with the SEEDS_MAIN of each dataset's
# pemsXX_run.sh (so this column compares 1-to-1 with the existing Retrain
# column in tables/main_table_full.tex).
# ===========================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# Defragment CUDA allocator. STNORM/STAEFORMER produce large intermediate
# activations whose size scales with N; on PEMS04/05 end-year (N≥3000) the
# default caching allocator leaves multi-GB of reserved-but-unallocated
# memory and OOMs on a single ~2GB request. expandable_segments=True merges
# these holes. See https://pytorch.org/docs/stable/notes/cuda.html.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Force line-buffered stdout/stderr so `tail -f <log>` shows output in real
# time instead of in 8-64 KiB bursts. Python switches to block-buffering by
# default when stdout is redirected to a file (which is what `>> $log` does).
export PYTHONUNBUFFERED=1

DATASETS=${DATASETS:-"PEMS03 PEMS04 PEMS05 PEMS06 PEMS07 PEMS08 PEMS10 PEMS11 PEMS12"}
SEEDS=${SEEDS:-"42 43 44 45 46"}
METHODS=${METHODS:-"gwn stid itransformer dlinear stnorm staeformer"}
GPU=${GPU:-0}

# ---------------------------------------------------------------------------
# Background mode: NOHUP=1 -> tee output to run_logs/extra_baselines_<tag>.log
# The log filename bakes in the methods (or "all" if the full default set)
# plus GPU, so `ls run_logs/` tells you what each file is at a glance.
# Pass RUN_TAG to override entirely if you want a custom name.
# ---------------------------------------------------------------------------
DEFAULT_METHODS="gwn stid itransformer dlinear stnorm staeformer"
if [[ "$METHODS" == "$DEFAULT_METHODS" ]]; then
    METHOD_TAG="all"
else
    METHOD_TAG=$(echo "$METHODS" | tr ' ' '_' | sed 's/^_//;s/_$//')
fi

if [[ "${NOHUP:-0}" == "1" && -z "${EAC_BG:-}" ]]; then
    mkdir -p run_logs
    TAG="${RUN_TAG:-${METHOD_TAG}_gpu${GPU}}"
    LOG_FILE="run_logs/extra_baselines_${TAG}_$(date +%Y%m%d_%H%M%S).log"
    echo "[nohup] backgrounding to $LOG_FILE"
    EAC_BG=1 nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "[nohup] PID=$BG_PID"
    echo "[nohup] tail -f $LOG_FILE"
    exit 0
fi

echo "[run] DATASETS=$DATASETS"
echo "[run] METHODS=$METHODS"
echo "[run] SEEDS=$SEEDS"
echo "[run] GPU=$GPU"

for ds in $DATASETS; do
    low=$(echo "$ds" | tr 'A-Z' 'a-z')
    echo ""
    echo "############################################################"
    echo "### Dataset = $ds"
    echo "############################################################"

    for m in $METHODS; do
        conf="conf/${ds}/retrain_${m}_${low}.json"
        if [[ ! -f "$conf" ]]; then
            echo "  [skip] missing config: $conf"
            continue
        fi
        echo "---------- [$ds] retrain backbone=$m ----------"
        for seed in $SEEDS; do
            python main.py --conf "$conf" --gpuid "$GPU" --seed "$seed"
        done
    done
done

echo ""
echo "==================== EXTRA BASELINES ALL DONE ===================="
