#!/bin/bash
#SBATCH --job-name=obs_regen
#SBATCH --partition=main
#SBATCH --array=1-96%12
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/scratch/obs_logs/obs_%a.out
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/obs_cmds.txt)
TAG="${LINE%%$'\t'*}"
OVR="${LINE#*$'\t'}"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_obs_results.txt
export OMP_NUM_THREADS=8
cd /path/to/repos/protection-eval-framework
echo "TASK $SLURM_ARRAY_TASK_ID: $TAG :: $OVR"
.venv/bin/python /path/to/scratch/run_obs.py "$TAG" $OVR
