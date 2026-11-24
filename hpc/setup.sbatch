#!/bin/bash
#SBATCH --job-name=nm-setup
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=setup-%j.out

# Run from the repository root:
#   sbatch setup.sbatch                     # Qwen3-8B (tested default)
#   MODEL=qwen4 sbatch setup.sbatch
#   MODEL=deepseek8 sbatch setup.sbatch
#   MODEL=moonlight16 sbatch setup.sbatch
# Download one model at a time. Re-running setup keeps existing downloads.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repo root}"
ROOT="$PWD"
MODEL="${MODEL:-qwen8}"
case "$MODEL" in
  qwen4)      REPO="Qwen/Qwen3-4B" ;;
  qwen8)      REPO="Qwen/Qwen3-8B" ;;
  deepseek8)  REPO="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ;;
  moonlight16) REPO="moonshotai/Moonlight-16B-A3B-Instruct" ;;
  *) echo "Unknown MODEL=$MODEL (use qwen4, qwen8, deepseek8, moonlight16)" >&2; exit 1 ;;
esac

# Keep package/download caches out of the limited home directory.
export CONDA_PKGS_DIRS="$ROOT/.runtime/conda-pkgs"
export PIP_CACHE_DIR="$ROOT/.runtime/pip"
export HF_HOME="$ROOT/.runtime/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$ROOT/models"
ENV="$ROOT/.runtime/env"

module load anaconda3/2024.06
if [ ! -x "$ENV/bin/python" ]; then
  conda create -y --prefix "$ENV" --override-channels -c conda-forge python=3.10 pip
fi
PY="$ENV/bin/python"

# Same PyTorch/Transformers versions that worked for Qwen3-8B on Explorer.
# Python 3.10 helps with Moonlight's older custom-code requirements.
if [ ! -f "$ENV/.neuromesh-ready" ]; then
  "$PY" -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu118
  "$PY" -m pip install \
    'transformers==4.57.6' 'accelerate==1.15.0' \
    'safetensors==0.8.0' 'huggingface_hub==0.36.2' \
    sentencepiece einops tiktoken
  "$PY" -m pip check
  touch "$ENV/.neuromesh-ready"
fi

# Model files live in models/<choice>; HF caches live in .runtime/huggingface.
# The download can be resumed if a job times out or connection drops.
echo "Downloading $REPO to $ROOT/models/$MODEL"
"$PY" - "$REPO" "$ROOT/models/$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, destination = sys.argv[1:]
snapshot_download(repo_id=repo, local_dir=destination)
print("Download complete:", destination)
PY

echo "Ready: MODEL=$MODEL sbatch run.sbatch"
