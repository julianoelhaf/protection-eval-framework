#!/bin/bash
#SBATCH --job-name=rt_regen
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/scratch/rt_logs/rt_%j.out
set -f
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export RUN_OUT=/path/to/scratch/run_runtime_results.txt
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
cd /path/to/repos/protection-eval-framework
echo "RUNTIME NODE: $(hostname) cpus=${SLURM_CPUS_PER_TASK} $(date)"
run () { .venv/bin/python /path/to/scratch/run_runtime.py "$@"; }
# FC (event_type) -- default (reference) hyperparameters
run rt_fc_mlp_W20 training.target_label=event_type model.model_name=mlp_classifier                 window_extraction.window_length=0.020
run rt_fc_gb_W20  training.target_label=event_type model.model_name=hist_gradient_boosting_classifier window_extraction.window_length=0.020
run rt_fc_mlp_W50 training.target_label=event_type model.model_name=mlp_classifier                 window_extraction.window_length=0.050
run rt_fc_gb_W50  training.target_label=event_type model.model_name=hist_gradient_boosting_classifier window_extraction.window_length=0.050
# FL (y_fault_location)
run rt_fl_mlp_W20 training.target_label=y_fault_location model.model_name=mlp_regressor                 window_extraction.window_length=0.020
run rt_fl_gb_W20  training.target_label=y_fault_location model.model_name=hist_gradient_boosting_regressor window_extraction.window_length=0.020
run rt_fl_mlp_W50 training.target_label=y_fault_location model.model_name=mlp_regressor                 window_extraction.window_length=0.050
run rt_fl_gb_W50  training.target_label=y_fault_location model.model_name=hist_gradient_boosting_regressor window_extraction.window_length=0.050
echo "RUNTIME DONE $(date)"
