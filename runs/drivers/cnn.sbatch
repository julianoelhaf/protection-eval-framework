#!/bin/bash
#SBATCH --job-name=cnn_base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --array=1-4%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/scratch/cnn_logs/cnn_%a.out
# 1D-CNN baseline, one array task per (task,window). Runs the REPO harness copy
# (so CFG_DIR resolves via __file__) in the py_dl conda env, with fcl_psp on
# PYTHONPATH + the wandb/tabulate shims. GPU requested; the harness logs device=.
set -f
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /path/to/scratch/cnn_joblist.txt)
IFS='|' read -r TAG TGT WL <<< "$LINE"
REPO=/path/to/repos/protection-eval-framework
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export CNN_OUT=/path/to/scratch/cnn_results.txt
export PYTHONPATH=$REPO/src:$REPO/runs/harnesses/_shims
cd $REPO
echo "TASK $SLURM_ARRAY_TASK_ID $TAG $TGT $WL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
/path/to/conda/envs/py_dl/bin/python $REPO/runs/harnesses/cnn_baseline.py "$TAG" "$TGT" "$WL"
