#!/bin/bash
#SBATCH --job-name=stride
#SBATCH --partition=main
#SBATCH --array=1-36%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/scratch/stride_logs/stride_%a.out
# Stride sensitivity sweep: 2 tasks x 2 models x windows{10,20,50}ms x strides{5,10,20,50}ms
# (stride <= window) = 36 runs, reusing the general run_fc.py harness (handles FC + FL).
# Requires the (W,S) window caches built by build_stride_windows.sbatch.
set -f
REPO=/path/to/repos/protection-eval-framework
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$REPO/runs/drivers/stride_joblist.txt")
TAG="${LINE%%|*}"; OV="${LINE#*|}"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_stride_results.txt
cd "$REPO"
echo "TASK $SLURM_ARRAY_TASK_ID TAG=$TAG OV=$OV"
.venv/bin/python runs/harnesses/run_fc.py "$TAG" $OV
