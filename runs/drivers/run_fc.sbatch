#!/bin/bash
#SBATCH --job-name=run_fc
#SBATCH --partition=main
#SBATCH --array=1-29%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/scratch/run_logs/run_%a.out
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/run_joblist.txt)
TAG="${LINE%%|*}"; OV="${LINE#*|}"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_fc_results.txt
cd /path/to/repos/protection-eval-framework
echo "TASK $SLURM_ARRAY_TASK_ID TAG=$TAG OV=$OV"
.venv/bin/python /path/to/scratch/run_fc.py "$TAG" $OV
