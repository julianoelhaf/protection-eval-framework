#!/bin/bash
#SBATCH --job-name=moment
#SBATCH --partition=main
#SBATCH --gres=gpu:1
# Exclude Pascal/older GPUs (titanxp/gtx1080/gtx1080ti = sm_61) incompatible with torch 2.8+cu128,
# and the q8000 nodes (lme170/171) whose 60 GB RAM is too tight for the FC probe peak. Leaves the
# q5000/q6000 (sm_75, ~90 GB RAM) nodes.
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --array=1-4%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/scratch/moment_logs/moment_%a.out
# Pre-trained transformer baseline (MOMENT-1-large) — frozen embeddings -> linear probe + MLP head.
# Runs in the py_tsfm conda env (py_dl clone + momentfm/transformers). Weights are pre-downloaded to
# HF_HOME; HF_HUB_OFFLINE=1 avoids per-job network. Reuses run_fc.py's data/CV pipeline conventions.
set -f
REPO=/path/to/repos/protection-eval-framework
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$REPO/runs/drivers/moment_joblist.txt")
IFS='|' read -r TAG TGT WL <<< "$LINE"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export MOMENT_OUT=${MOMENT_OUT:-/path/to/scratch/moment_results.txt}
export MOMENT_BSZ=${MOMENT_BSZ:-48}
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export https_proxy=http://proxy.rrze.uni-erlangen.de:80
export PYTHONPATH=$REPO/src:$REPO/runs/harnesses/_shims
cd "$REPO"
echo "TASK $SLURM_ARRAY_TASK_ID $TAG $TGT $WL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
/path/to/conda/envs/py_tsfm/bin/python "$REPO/runs/harnesses/moment_baseline.py" "$TAG" "$TGT" "$WL"
