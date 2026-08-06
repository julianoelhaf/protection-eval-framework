#!/bin/bash
#SBATCH --job-name=ij_fcexp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=88G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/scratch/slurm_fcexp_%j.log
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export WINDIR=/path/to/scratch/win_fc2
PY=/path/to/repos/protection-eval-framework/.venv/bin/python
REPO=/path/to/repos/protection-eval-framework
WORK=/path/to/scratch
cd "$REPO" || exit 1
echo "START ij_fcexp on $(hostname) $(date)"
COMMON="window_extraction.windows_local_dir=$WINDIR tracking.mode=disabled hydra.output_subdir=null"

# --- FC conventional baseline (phase selector; thresholds fit on train folds) ---
for wl in 0.020 0.050; do
  echo "BASELINE FC W=$wl $(date)"
  "$PY" "$REPO/src/fcl_psp/baselines/run_conventional_baselines.py" \
    training.target_label=event_type window_extraction.window_length="$wl" +baseline.task=fc \
    hydra.run.dir="$WORK/h_fc_${wl//./p}" $COMMON || echo "FAIL fc baseline $wl"
done

# --- FC Experiment B perturbation (train-clean/test-perturbed): GB first, MLP last ---
for wl in 0.020 0.050; do
  for mdl in hist_gradient_boosting_classifier mlp_classifier; do
    echo "PERT FC $mdl W=$wl $(date)"
    "$PY" "$REPO/src/fcl_psp/models/run_perturbation_eval.py" \
      training.target_label=event_type model.model_name="$mdl" \
      window_extraction.window_length="$wl" \
      hydra.run.dir="$WORK/h_pfc_${mdl}_${wl//./p}" $COMMON || echo "FAIL pert fc $mdl $wl"
  done
done
echo "J3_FCEXP_DONE $(date)"
