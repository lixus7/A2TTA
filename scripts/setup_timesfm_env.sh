#!/bin/bash
# =============================================================================
# One-time setup for the TimesFM zero-shot baseline. RUN THIS ON A LOGIN NODE
# (it needs internet to install timesfm and download the checkpoint).
#
#   bash scripts/setup_timesfm_env.sh
#
# It is idempotent: re-running re-uses the existing env and cached checkpoint.
#
# What it does:
#   1. Creates a dedicated conda env `timesfm` (does NOT touch your stg/tsfm envs).
#   2. Installs timesfm[torch] + huggingface_hub (+ tqdm/numpy used by utils/).
#      timesfm's [torch] extra requires python 3.11 and pulls a CUDA torch wheel.
#   3. Verifies the CUDA torch wheel supports the target GPU arch (H200 = sm_90).
#   4. Picks the HuggingFace checkpoint (default: TimesFM 2.5, repo
#      google/timesfm-2.5-200m-pytorch; override via FORCE_REPO).
#   5. Pre-downloads that checkpoint into a PERSISTENT cache on a persistent scratch disk so
#      offline compute nodes can load it with HF_HUB_OFFLINE=1.
#   6. Regenerates conf/PEMS*/timesfm_*.json so repo_id matches the resolved repo.
# =============================================================================
set -euo pipefail

# ---- knobs -----------------------------------------------------------------
ENV_NAME=${ENV_NAME:-timesfm}
PY_VER=${PY_VER:-3.11}          # timesfm[torch] extra is gated to python 3.11
REPO_ROOT=${REPO_ROOT:-.}
# Persistent HF cache shared by setup (online) and the HPC job (offline).
export HF_HOME=${HF_HOME:-./hf_cache}
CONDA_SH=${CONDA_SH:-$CONDA
# IMPORTANT: PyPI `timesfm` (<=1.3.0) only ships the CLASSIC API (TimesFm /
# TimesFmHparams) and CANNOT load the 2.5 checkpoint. The TimesFM 2.5 API
# (TimesFM_2p5_200M_torch / ForecastConfig / from_pretrained) lives only on
# GitHub master, so we install from source to get the 2.5 model.
# To use the classic 2.0 model instead, set TIMESFM_PIP='timesfm[torch]' and
# FORCE_REPO='google/timesfm-2.0-500m-pytorch'.
TIMESFM_PIP=${TIMESFM_PIP:-'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'}
# Default to the TimesFM 2.5 PyTorch checkpoint. Set FORCE_REPO='' to instead
# auto-detect (classic 2.0 if the 2.5 API is absent).
FORCE_REPO=${FORCE_REPO:-'google/timesfm-2.5-200m-pytorch'}
# GPU arch the env must support (H200 = Hopper = sm_90).
NEED_ARCH=${NEED_ARCH:-sm_90}

echo "[setup] env=$ENV_NAME  HF_HOME=$HF_HOME  pip='$TIMESFM_PIP'"
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
echo "[setup] installing $TIMESFM_PIP + huggingface_hub ..."
pip install --upgrade pip
pip install "$TIMESFM_PIP" huggingface_hub
# utils/data_convert.py needs tqdm; utils/* need numpy (timesfm pulls torch+numpy)
pip install tqdm

python - <<'PY'
import timesfm, sys
print("[setup] timesfm version:", getattr(timesfm, "__version__", "?"))
PY

# ---- 3. verify CUDA torch supports the target GPU (H200 = sm_90) -----------
echo "[setup] verifying torch CUDA build supports $NEED_ARCH (H200) ..."
NEED_ARCH="$NEED_ARCH" python - <<'PY'
import os, torch
arch = os.environ["NEED_ARCH"]
print("[setup] torch", torch.__version__, "cuda build:", torch.version.cuda)
archs = torch.cuda.get_arch_list()
print("[setup] torch arch_list:", archs)
if torch.version.cuda is None:
    raise SystemExit(f"[setup] ERROR: torch is CPU-only — will not run on H200. "
                     f"Reinstall with a CUDA wheel, e.g.\n"
                     f"  pip install torch --index-url https://download.pytorch.org/whl/cu124")
if not any(arch in a for a in archs):
    raise SystemExit(f"[setup] ERROR: torch wheel lacks {arch} (H200) support; archs={archs}. "
                     f"Reinstall a cu12x torch wheel that includes {arch}.")
print(f"[setup] OK: torch CUDA build includes {arch} (H200-ready).")
PY

# ---- 4. choose checkpoint repo --------------------------------------------
if [[ -n "$FORCE_REPO" ]]; then
    RESOLVED_REPO="$FORCE_REPO"
    echo "[setup] using forced checkpoint repo: $RESOLVED_REPO"
    # sanity: warn if 2.5 repo requested but the 2.5 API is not importable
    python - "$RESOLVED_REPO" <<'PY' || true
import sys, timesfm
repo = sys.argv[1]
has25 = any(a.startswith("TimesFM_2p5") for a in dir(timesfm)) and hasattr(timesfm, "ForecastConfig")
if "2.5" in repo and not has25:
    print("[setup] WARNING: repo is 2.5 but installed timesfm lacks the 2.5 API "
          "(TimesFM_2p5_* / ForecastConfig). Upgrade timesfm or set FORCE_REPO to a 2.0 repo.")
PY
else
    RESOLVED_REPO=$(python - <<'PY'
import timesfm
is_25 = any(a.startswith("TimesFM_2p5") for a in dir(timesfm)) and hasattr(timesfm, "ForecastConfig")
print("google/timesfm-2.5-200m-pytorch" if is_25 else "google/timesfm-2.0-500m-pytorch")
PY
)
    echo "[setup] auto-detected checkpoint repo: $RESOLVED_REPO"
fi

# ---- 5. pre-download checkpoint into persistent cache ----------------------
echo "[setup] downloading $RESOLVED_REPO into $HF_HOME (one-time) ..."
HF_HOME="$HF_HOME" python - "$RESOLVED_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = snapshot_download(repo_id=repo)
print("[setup] checkpoint cached at:", path)
PY

# ---- 6. regenerate configs so repo_id matches ------------------------------
echo "[setup] regenerating conf/PEMS*/timesfm_*.json with repo_id=$RESOLVED_REPO"
TIMESFM_REPO="$RESOLVED_REPO" python scripts/gen_timesfm_configs.py

echo
echo "[setup] DONE."
echo "  conda env      : $ENV_NAME"
echo "  HF cache       : $HF_HOME   (set HF_HOME on compute nodes, with HF_HUB_OFFLINE=1)"
echo "  checkpoint     : $RESOLVED_REPO"
echo
echo "Quick smoke test on this node (1 dataset, first 2 years, GPU 0 if available):"
echo "  conda activate $ENV_NAME"
echo "  HF_HOME=$HF_HOME python timesfm_main.py --conf conf/PEMS03/timesfm_pems03.json \\"
echo "      --gpuid 0 --seed 51 --fast_dev_run 2 --csv_path run_logs/timesfm_smoke.csv"
