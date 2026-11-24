#!/bin/bash
#SBATCH --job-name=nm-agents
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=run-%j.out

# Submit from the hpc/ directory inside the cloned repository:
#   sbatch run.sbatch                                  # Qwen3-8B seven-agent prototype (N=7, 1 round)
#   MODEL=qwen4 sbatch run.sbatch
#   MODEL=deepseek8 sbatch run.sbatch
#   MODEL=moonlight16 sbatch --mem=48G run.sbatch    # experimental, may need H200
#   MODEL=qwen8 sbatch -p sharing --gres=gpu:a100:1 run.sbatch
#   MODEL=moonlight16 sbatch -p gpu --gres=gpu:h200:1 --mem=48G -t 04:00:00 run.sbatch
#   SCRIPT=example_agents.py sbatch run.sbatch          # old Solver/Critic demo
#   N_AGENTS=7 ROUNDS=10 sbatch run.sbatch             # ten-round prototype
# The partition/GPU/time overrides are scheduler requests, not guarantees.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from hpc/}"
ROOT="$PWD"
export MODEL="${MODEL:-qwen8}"
SCRIPT="${SCRIPT:-prototype_network.py}"
export N_AGENTS="${N_AGENTS:-7}"
export ROUNDS="${ROUNDS:-1}"
export BASELINE="${BASELINE:-1}"
export HF_HOME="$ROOT/.runtime/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ ! -x "$ROOT/.runtime/env/bin/python" ]; then
  echo "Missing environment; first run: sbatch setup.sbatch" >&2
  exit 1
fi
if [ ! -f "$ROOT/models/$MODEL/config.json" ]; then
  echo "Missing model $MODEL; first run: MODEL=$MODEL sbatch setup.sbatch" >&2
  exit 1
fi

module load anaconda3/2024.06
echo "Job $SLURM_JOB_ID | Node $(hostname -s) | MODEL=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f "$ROOT/$SCRIPT" ]; then
  echo "Missing Python script: $ROOT/$SCRIPT" >&2
  exit 1
fi
echo "Python script: $SCRIPT"
exec "$ROOT/.runtime/env/bin/python" "$ROOT/$SCRIPT"
