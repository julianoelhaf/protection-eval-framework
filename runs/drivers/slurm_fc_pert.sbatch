#!/bin/bash
#SBATCH --job-name=ij_fcpert
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=88G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/scratch/slurm_fcpert_%j.log
# Re-run of FC Experiment-B perturbation (J3's runs crashed on the pre-fix config bug).
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export WINDIR=/path/to/scratch/win_fc2
PY=/path/to/repos/protection-eval-framework/.venv/bin/python
REPO=/path/to/repos/protection-eval-framework
WORK=/path/to/scratch
cd "$REPO" || exit 1
echo "START ij_fcpert on $(hostname) $(date)"
COMMON="window_extraction.windows_local_dir=$WINDIR tracking.mode=disabled hydra.output_subdir=null"
for wl in 0.020 0.050; do
  for mdl in hist_gradient_boosting_classifier mlp_classifier; do
    echo "PERT FC $mdl W=$wl $(date)"
    "$PY" "$REPO/src/fcl_psp/models/run_perturbation_eval.py" \
      training.target_label=event_type model.model_name="$mdl" \
      window_extraction.window_length="$wl" \
      hydra.run.dir="$WORK/h_pfc2_${mdl}_${wl//./p}" $COMMON || echo "FAIL pert fc $mdl $wl"
  done
done
echo "FCPERT_RERUN_DONE $(date)"
