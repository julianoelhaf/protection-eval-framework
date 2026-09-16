#!/bin/bash
#SBATCH --job-name=tim10
#SBATCH --partition=main
#SBATCH --array=1-8%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/scratch/tim10_logs/tim10_%a.out
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/tim10_cmds.txt)
TAG="${LINE%%$'\t'*}"; OVR="${LINE#*$'\t'}"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_tim10_results.txt
export OMP_NUM_THREADS=8
cd /path/to/repos/protection-eval-framework
echo "TASK $SLURM_ARRAY_TASK_ID: $TAG :: $OVR"
.venv/bin/python /path/to/scratch/run_fc.py "$TAG" $OVR
