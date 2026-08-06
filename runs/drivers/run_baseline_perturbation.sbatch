#!/bin/bash
#SBATCH --job-name=blp_fidelity
#SBATCH --partition=main
#SBATCH --array=1-6%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/scratch/blp_logs/blp_%a.out
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" runs/drivers/blp_joblist.txt)
IFS='|' read -r TASK TGT WL <<< "$LINE"
export OMP_NUM_THREADS=8
cd /path/to/repos/protection-eval-framework
echo "BLP task=$TASK target=$TGT wl=$WL $(date)"
.venv/bin/python -m fcl_psp.baselines.run_baseline_perturbation \
  window_extraction.windows_local_dir=/path/to/datasets/PROTECT-90/windows/PROTECT-90 \
  tracking.mode=disabled +baseline.task="$TASK" training.target_label="$TGT" window_extraction.window_length="$WL"
