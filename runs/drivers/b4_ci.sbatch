#!/bin/bash
#SBATCH --job-name=b4_ci
#SBATCH --partition=main
#SBATCH --array=1-8%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/scratch/b4_ci_logs/b4_%a.out
# Generalization-under-R_f-shift WITH episode-level bootstrap 95% CI on the
# shifted-test metric. Point estimates reproduce the committed b4 run (seed-fixed);
# adds boot_std + ci95. Reuses the same joblist as b4.sbatch.
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/b4_joblist.txt)
IFS='|' read -r TAG TGT MDL WL DIR <<< "$LINE"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export B4_OUT=/path/to/scratch/b4_ci_results.txt
cd /path/to/repos/protection-eval-framework
echo "TASK $SLURM_ARRAY_TASK_ID $TAG $TGT $MDL $WL $DIR"
.venv/bin/python /path/to/scratch/b4_generalization.py "$TAG" "$TGT" "$MDL" "$WL" "$DIR"
