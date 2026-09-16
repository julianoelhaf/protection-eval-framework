#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=main
#SBATCH --array=1-108%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=23:59:00
#SBATCH --output=/path/to/scratch/ablation_logs/ablation_%a.out
# Hyperparameter-ablation regeneration (Workstream B tail): FC/FL x 20/50 ms, per-model knob grid
# + per-cell default (baseline). 108 runs via the general run_fc.py harness (handles FC + FL).
# The committed FC-20 grid (run_fc_results.txt abl_*) is re-run here under a unified tag scheme so the
# aggregation reads one consistent source (reports/runs/run_ablation_results.txt).
set -f
REPO=/path/to/repos/protection-eval-framework
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$REPO/runs/drivers/ablation_joblist.txt")
TAG="${LINE%%|*}"; OV="${LINE#*|}"
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_ablation_results.txt
cd "$REPO"
echo "TASK $SLURM_ARRAY_TASK_ID TAG=$TAG OV=$OV"
.venv/bin/python runs/harnesses/run_fc.py "$TAG" $OV
rm -f "$WINDIR/X_fo_run_${TAG}.raw"   # reclaim the per-tag fault-only scratch memmap (FL runs only)
