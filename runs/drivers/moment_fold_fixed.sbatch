#!/bin/bash
#SBATCH --job-name=moment_fix
#SBATCH --partition=main
#SBATCH --gres=gpu:1
# Exclude Pascal/older GPUs (titanxp/gtx1080/gtx1080ti = sm_61) incompatible with torch 2.8+cu128,
# and the q8000 nodes (lme170/171) whose 60 GB RAM is too tight for the FC probe peak. Leaves the
# q5000/q6000 (sm_75, ~90 GB RAM) nodes.
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171
#SBATCH --array=1-20%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/scratch/moment_fold_fixed_logs/%A_%a.out
# Corrected MOMENT-1-large baseline: ONE outer fold per array task. Each job fits a raw-window
# StandardScaler on that fold's training episodes only, produces fold-specific train/test embeddings
# with the frozen backbone, and writes one atomic JSON to MOMENT_PART_OUT. Aggregation into the
# five-fold result lines is a separate step (aggregate_moment_results.py). Runs in the py_tsfm conda
# env; weights are pre-downloaded to HF_HOME and HF_HUB_OFFLINE=1 avoids per-job network.
set -euo pipefail
REPO=/path/to/repos/protection-eval-framework
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$REPO/runs/drivers/moment_fold_fixed_joblist.txt")
IFS='|' read -r TAG TARGET WINDOW FOLD <<< "$LINE"

export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export MOMENT_BSZ=${MOMENT_BSZ:-48}
export HF_HOME=/path/to/hf_cache
export HF_HUB_OFFLINE=1
export https_proxy=http://proxy.rrze.uni-erlangen.de:80
export PYTHONPATH=$REPO/src:$REPO/runs/harnesses/_shims
export MOMENT_PART_OUT="${MOMENT_PART_DIR}/${TAG}_fold${FOLD}.json"
cd "$REPO"

echo "TASK=$SLURM_ARRAY_TASK_ID"
echo "TAG=$TAG"
echo "TARGET=$TARGET"
echo "WINDOW=$WINDOW"
echo "OUTER_FOLD=$FOLD"
echo "PART_OUT=$MOMENT_PART_OUT"
# Informational only; must never abort the job (nvidia-smi can exit non-zero and, under
# pipefail, SIGPIPE from `head` closing the pipe would otherwise kill the run).
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

/path/to/conda/envs/py_tsfm/bin/python \
  "$REPO/runs/harnesses/moment_baseline.py" \
  "$TAG" "$TARGET" "$WINDOW" "$FOLD"
