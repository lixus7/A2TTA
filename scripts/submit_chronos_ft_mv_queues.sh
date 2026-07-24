#!/bin/bash
set -Eeuo pipefail

# Submit the multivariate Chronos array across one or two arbitrary PBS queues.
# Queue names and local capacities are supplied at runtime, for example:
#   HPC_QUEUE_A=<queue> HPC_QUEUE_A_LANES=4 \
#   HPC_QUEUE_B=<queue> HPC_QUEUE_B_LANES=2 \
#   bash scripts/submit_chronos_ft_mv_queues.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${A2TTA_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
WORKER=${A2TTA_WORKER:-"$REPO/scripts/hpc_chronos_ft_mv_array.pbs"}
LOG_DIR=${A2TTA_LOG_DIR:-"$REPO/run_logs/chronos_ft_mv_pbs"}

: "${HPC_QUEUE_A:?Set HPC_QUEUE_A to the primary PBS queue name.}"
HPC_QUEUE_B=${HPC_QUEUE_B:-}
HPC_QUEUE_A_LANES=${HPC_QUEUE_A_LANES:-1}
HPC_QUEUE_B_LANES=${HPC_QUEUE_B_LANES:-1}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

if ! is_positive_integer "$HPC_QUEUE_A_LANES"; then
    echo "HPC_QUEUE_A_LANES must be a positive integer." >&2
    exit 2
fi
if [[ -n "$HPC_QUEUE_B" ]] && ! is_positive_integer "$HPC_QUEUE_B_LANES"; then
    echo "HPC_QUEUE_B_LANES must be a positive integer." >&2
    exit 2
fi

if [[ -z "$HPC_QUEUE_B" ]]; then
    HPC_QUEUE_B_LANES=0
fi
TOTAL_LANES=$((HPC_QUEUE_A_LANES + HPC_QUEUE_B_LANES))

if [[ ! -f "$WORKER" ]]; then
    echo "The configured worker script is missing." >&2
    exit 3
fi
if [[ ${DRY_RUN:-0} != 1 ]] && ! command -v qsub >/dev/null 2>&1; then
    echo "qsub is unavailable" >&2
    exit 4
fi

mkdir -p "$LOG_DIR"

# The worker creates three seed tasks for every discovered dataset-year file.
if [[ -n "${A2TTA_TOTAL_TASKS:-}" ]]; then
    if ! is_positive_integer "$A2TTA_TOTAL_TASKS"; then
        echo "A2TTA_TOTAL_TASKS must be a positive integer." >&2
        exit 5
    fi
    TOTAL_TASKS=$A2TTA_TOTAL_TASKS
else
    DATASETS=(pems03 pems04 pems05 pems06 pems07 pems08 pems10 pems11 pems12 tfnsw)
    TOTAL_TASKS=0
    shopt -s nullglob
    for dataset in "${DATASETS[@]}"; do
        if [[ "$dataset" == tfnsw ]]; then
            raw_dir="$REPO/../tfnsw/RawData"
        else
            raw_dir="$REPO/../xxltrafficdata/preprocessed/$dataset/RawData"
        fi
        raw_files=("$raw_dir"/*.npz)
        TOTAL_TASKS=$((TOTAL_TASKS + ${#raw_files[@]} * 3))
    done
    shopt -u nullglob
fi

if (( TOTAL_TASKS < 1 )); then
    echo "No tasks were discovered; check A2TTA_REPO or set A2TTA_TOTAL_TASKS." >&2
    exit 5
fi

lane_task_count() {
    local lane_offset=$1
    local lane_count=$2
    local full_cycles=$((TOTAL_TASKS / TOTAL_LANES))
    local remainder=$((TOTAL_TASKS % TOTAL_LANES))
    local extra=$((remainder - lane_offset))
    if (( extra < 0 )); then
        extra=0
    elif (( extra > lane_count )); then
        extra=$lane_count
    fi
    printf '%s\n' $((full_cycles * lane_count + extra))
}

QUEUE_A_TASKS=$(lane_task_count 0 "$HPC_QUEUE_A_LANES")
QUEUE_B_TASKS=$(lane_task_count "$HPC_QUEUE_A_LANES" "$HPC_QUEUE_B_LANES")

echo "TOTAL_TASKS=$TOTAL_TASKS TOTAL_LANES=$TOTAL_LANES"
echo "QUEUE_A_TASKS=$QUEUE_A_TASKS QUEUE_A_LANES=$HPC_QUEUE_A_LANES LANE_OFFSET=0"
if [[ -n "$HPC_QUEUE_B" ]]; then
    echo "QUEUE_B_TASKS=$QUEUE_B_TASKS QUEUE_B_LANES=$HPC_QUEUE_B_LANES LANE_OFFSET=$HPC_QUEUE_A_LANES"
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi

worker_vars="TOTAL_LANES=$TOTAL_LANES,A2TTA_REPO=$REPO"
if [[ -n "${A2TTA_PYTHON:-}" ]]; then
    worker_vars+=",A2TTA_PYTHON=$A2TTA_PYTHON"
fi
if [[ -n "${CHRONOS_MODEL:-}" ]]; then
    worker_vars+=",CHRONOS_MODEL=$CHRONOS_MODEL"
fi
if [[ -n "${CHRONOS_MODEL_LABEL:-}" ]]; then
    worker_vars+=",CHRONOS_MODEL_LABEL=$CHRONOS_MODEL_LABEL"
fi
if [[ -n "${HF_HOME:-}" ]]; then
    worker_vars+=",HF_HOME=$HF_HOME"
fi

QUEUE_A_JOB=$(
    qsub \
        -N C2MV_A \
        -q "$HPC_QUEUE_A" \
        -J "1-${QUEUE_A_TASKS}%${HPC_QUEUE_A_LANES}" \
        -v "$worker_vars,QUEUE_LANES=$HPC_QUEUE_A_LANES,LANE_OFFSET=0" \
        -o "$LOG_DIR" \
        "$WORKER"
)

echo "QUEUE_A_JOB_ID=${QUEUE_A_JOB%%.*}"

if [[ -n "$HPC_QUEUE_B" ]]; then
    QUEUE_B_JOB=$(
        qsub \
            -N C2MV_B \
            -q "$HPC_QUEUE_B" \
            -J "1-${QUEUE_B_TASKS}%${HPC_QUEUE_B_LANES}" \
            -v "$worker_vars,QUEUE_LANES=$HPC_QUEUE_B_LANES,LANE_OFFSET=$HPC_QUEUE_A_LANES" \
            -o "$LOG_DIR" \
            "$WORKER"
    )
    echo "QUEUE_B_JOB_ID=${QUEUE_B_JOB%%.*}"
fi

echo "TOTAL_SUBMITTED_SHARDS=$((QUEUE_A_TASKS + QUEUE_B_TASKS))"
