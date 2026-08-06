#!/bin/bash
#SBATCH --job-name=ij_fl
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/path/to/scratch/slurm_fl_%j.log
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export WINDIR=/path/to/scratch/win_fl
export REPRO_OUT=/path/to/scratch/repro_fl.txt
PY=/path/to/repos/protection-eval-framework/.venv/bin/python
REPO=/path/to/repos/protection-eval-framework
WORK=/path/to/scratch
cd "$REPO" || exit 1
echo "START ij_fl on $(hostname) $(date)"
COMMON="window_extraction.windows_local_dir=$WINDIR tracking.mode=disabled hydra.output_subdir=null"

# --- FL reproduction (reference table): fast models first, MLP last ---
for wl in 0.020 0.050; do
  for mdl in ridge_regressor k_neighbors_regressor hist_gradient_boosting_regressor mlp_regressor; do
    echo "REPRO FL $mdl W=$wl $(date)"
    "$PY" "$WORK/smoke_reproduce.py" y_fault_location "$mdl" "$wl" || echo "FAIL repro $mdl $wl"
  done
done

# --- FL Experiment B perturbation (train-clean/test-perturbed): GB first, MLP last ---
for wl in 0.020 0.050; do
  for mdl in hist_gradient_boosting_regressor mlp_regressor; do
    echo "PERT FL $mdl W=$wl $(date)"
    "$PY" "$REPO/src/fcl_psp/models/run_perturbation_eval.py" \
      training.target_label=y_fault_location model.model_name="$mdl" \
      window_extraction.window_length="$wl" \
      hydra.run.dir="$WORK/h_pfl_${mdl}_${wl//./p}" $COMMON || echo "FAIL pert $mdl $wl"
  done
done
echo "J1_FL_DONE $(date)"
