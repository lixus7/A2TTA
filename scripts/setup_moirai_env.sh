#!/bin/bash
# =============================================================================
# One-time setup for the Moirai (uni2ts) baselines. RUN ON A LOGIN NODE
# (needs internet to pip-install uni2ts and download the checkpoints).
#
#   bash scripts/setup_moirai_env.sh
#
# Idempotent: re-running re-uses the existing env and cached checkpoints.
#
# What it does:
#   1. Creates a dedicated conda env `moirai` (does NOT touch stg/tsfm/chronos/
#      timesfm envs — uni2ts pins torch>=2.1,<2.5, incompatible with their torch).
#   2. Installs a CUDA torch wheel that supports H200 (sm_90) and is <2.5, then
#      uni2ts==2.0.0 (the PyPI release that ships Moirai-2.0) + tqdm (utils/ dep).
#   3. Verifies the torch CUDA build supports sm_90 (H200).
#   4. Pre-downloads BOTH checkpoints into a PERSISTENT cache on a persistent scratch disk so
#      offline compute nodes can load them with HF_HUB_OFFLINE=1:
#        - Salesforce/moirai-2.0-R-small   (Moirai-2.0, quantile decoder)
#        - Salesforce/moirai-moe-1.0-R-base (Moirai-MoE base, 0.9B)
#   5. Regenerates conf/PEMS*/moirai_*.json + moirai_ft_*.json.
# =============================================================================
set -euo pipefail

ENV_NAME=${ENV_NAME:-moirai}
PY_VER=${PY_VER:-3.11}              # uni2ts needs >=3.10
REPO_ROOT=${REPO_ROOT:-.}
export HF_HOME=${HF_HOME:-./hf_cache}
CONDA_SH=${CONDA_SH:-$CONDA
# uni2ts pins torch>=2.1,<2.5. torch 2.4.1 cu121 supports H200 (sm_90).
TORCH_PIP=${TORCH_PIP:-'torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121'}
UNI2TS_PIP=${UNI2TS_PIP:-'uni2ts==2.0.0'}     # first PyPI release with Moirai-2.0
NEED_ARCH=${NEED_ARCH:-sm_90}                 # H200 = Hopper
MODELS=${MODELS:-"Salesforce/moirai-2.0-R-small Salesforce/moirai-moe-1.0-R-base"}

echo "[setup] env=$ENV_NAME  HF_HOME=$HF_HOME"
echo "[setup] torch='$TORCH_PIP'  uni2ts='$UNI2TS_PIP'"
mkdir -p "$HF_HOME"
cd "$REPO_ROOT"

# ---- 1. conda env ----------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] creating conda env $ENV_NAME (python $PY_VER)"
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
else
    echo "[setup] conda env $ENV_NAME already exists"
fi
conda activate "$ENV_NAME"

# ---- 2. install ------------------------------------------------------------
pip install --upgrade pip
echo "[setup] installing torch ..."
# shellcheck disable=SC2086
pip install $TORCH_PIP
echo "[setup] installing uni2ts (+ tqdm for utils/data_convert) ..."
pip install "$UNI2TS_PIP" tqdm

python - <<'PY'
import uni2ts, torch
print("[setup] uni2ts:", getattr(uni2ts, "__version__", "?"), "torch:", torch.__version__)
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module       # noqa
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule  # noqa
print("[setup] Moirai2 + MoiraiMoE imports OK")
PY

# ---- 3. verify CUDA torch supports H200 (sm_90) ----------------------------
echo "[setup] verifying torch CUDA build supports $NEED_ARCH (H200) ..."
NEED_ARCH="$NEED_ARCH" python - <<'PY'
import os, torch
arch = os.environ["NEED_ARCH"]
print("[setup] torch", torch.__version__, "cuda build:", torch.version.cuda)
archs = torch.cuda.get_arch_list()
print("[setup] torch arch_list:", archs)
if torch.version.cuda is None:
    raise SystemExit("[setup] ERROR: torch is CPU-only — will not run on H200.")
if not archs:
    # get_arch_list() is empty on a GPU-less login node (CUDA can't init). The
    # cu121 torch 2.4.1 wheel ships sm_90; verify for real on the compute node.
    print(f"[setup] WARN: cannot introspect arch_list here (no GPU on login node); "
          f"cu12x wheel supports {arch}. Verify on the H200 compute node.")
elif not any(arch in a for a in archs):
    raise SystemExit(f"[setup] ERROR: torch wheel lacks {arch} (H200); archs={archs}.")
else:
    print(f"[setup] OK: torch CUDA build includes {arch} (H200-ready).")
PY

# ---- 4. pre-download checkpoints into persistent cache ---------------------
for repo in $MODELS; do
    echo "[setup] downloading $repo into $HF_HOME (one-time) ..."
    HF_HOME="$HF_HOME" python - "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo_id=repo)
print("[setup] cached:", repo, "->", path)
PY
done

# ---- 5. regenerate configs -------------------------------------------------
echo "[setup] regenerating conf/PEMS*/moirai_*.json ..."
python scripts/gen_moirai_configs.py
python scripts/gen_moirai_ft_configs.py

echo
echo "[setup] DONE."
echo "  conda env  : $ENV_NAME"
echo "  HF cache   : $HF_HOME   (set HF_HOME + HF_HUB_OFFLINE=1 on compute nodes)"
echo "  models     : $MODELS"
echo
echo "Smoke test (CPU ok, 1 dataset, first 2 years):"
echo "  conda activate $ENV_NAME"
echo "  HF_HOME=$HF_HOME python moirai_main.py --conf conf/PEMS03/moirai_moirai2_pems03.json \\"
echo "      --gpuid 0 --fast_dev_run 2 --max_samples 64 --csv_path run_logs/moirai_smoke.csv"
