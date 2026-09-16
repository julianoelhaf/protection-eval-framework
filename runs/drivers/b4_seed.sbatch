#!/bin/bash
#SBATCH --job-name=b4_seed
#SBATCH --partition=main
#SBATCH --array=1-24%6
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/scratch/b4_seed_logs/b4_%a.out
# Generalization-under-R_f-shift with a seed sweep: MLP cells x 5 seeds -> mean+/-std
# (sklearn MLP is non-deterministic run-to-run); GB cells x 1 (deterministic).
# Uses the pinned .venv (matches the reference/timing/obs runs). Bootstrap off.
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/b4_seed_joblist.txt)
IFS='|' read -r TAG TGT MDL WL DIR SEED <<< "$LINE"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export B4_OUT=/path/to/scratch/b4_seed_results.txt
cd /path/to/repos/protection-eval-framework
echo "TASK $SLURM_ARRAY_TASK_ID $TAG $TGT $MDL $WL $DIR seed=$SEED"
.venv/bin/python /path/to/scratch/b4_generalization.py "$TAG" "$TGT" "$MDL" "$WL" "$DIR" "$SEED"
