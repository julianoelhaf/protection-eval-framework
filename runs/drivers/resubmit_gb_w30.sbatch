#!/bin/bash
#SBATCH --job-name=run_gbW30
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/scratch/run_logs/run_gbW30_solo.out
set -f
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_fc_results.txt
cd /path/to/repos/protection-eval-framework
.venv/bin/python /path/to/scratch/run_fc.py tim_gb_W30 \
  training.target_label=event_type model.model_name=hist_gradient_boosting_classifier window_extraction.window_length=0.030
