#!/bin/bash
#SBATCH --job-name=ij_fcrepro
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=88G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/scratch/slurm_fcrepro_%j.log
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
export WINDIR=/path/to/scratch/win_fc
export REPRO_OUT=/path/to/scratch/repro_fc.txt
PY=/path/to/repos/protection-eval-framework/.venv/bin/python
WORK=/path/to/scratch
cd /path/to/repos/protection-eval-framework || exit 1
echo "START ij_fcrepro on $(hostname) $(date)"

# --- FC reproduction (11-class macro-F1): fast models first, MLP last (the 0.99 gate) ---
for wl in 0.020 0.050; do
  for mdl in ridge_classifier k_neighbors_classifier hist_gradient_boosting_classifier mlp_classifier; do
    echo "REPRO FC $mdl W=$wl $(date)"
    "$PY" "$WORK/smoke_reproduce.py" event_type "$mdl" "$wl" || echo "FAIL fcrepro $mdl $wl"
  done
done
echo "J2_FCREPRO_DONE $(date)"
