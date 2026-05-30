#!/bin/bash
#SBATCH --job-name=skyfusion-yolo
#SBATCH -A plgzzsn2026-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --output=logs/skyfusion-%j.out
#SBATCH --error=logs/skyfusion-%j.err

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load GCCcore/13.3.0
module load Python/3.12.3
python --version
pip install uv
export UV_CACHE_DIR="$SCRATCH/aero-detect/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$SCRATCH/aero-detect/.venv"
export KAGGLEHUB_CACHE="$SCRATCH/aero-detect/.cache/kagglehub"

module load CUDA/12.8.0

uv sync

uv run ./aerodetect/modeling/train.py +db=skyfusion
