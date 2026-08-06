#!/bin/bash
#SBATCH --job-name=fc_falsepos
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/repos/protection-eval-framework/runs/logs/security/fc_falsepos_%j.out
# Real no-fault false-positive / nuisance-trip rate for the FC task (PROTECT-90).
# One 5-fold sklearn-MLP CPU run of the *committed* pipeline (fcl_psp.models.run_model),
# reference FC config: event_type / mlp_classifier / window 0.020 s (the paper CM figure).
# run_model.main() ALWAYS calls save_offline_outputs() -> persists per-window OOF
# predictions (y_true, y_pred, status, fold) to oof_predictions.parquet; no flag needed.
# The OOF dump is routed to cluster scratch (hydra.run.dir + hydra.job.chdir=true) so the
# git working tree stays clean. A second step parses it for the FP / nuisance-trip rates.
set -eo pipefail
set -f  # keep Hydra bracket-free overrides safe from globbing
REPO=/path/to/repos/protection-eval-framework
export WINDIR=/path/to/datasets/PROTECT-90/windows/PROTECT-90
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
OUTROOT=/path/to/scratch/fc_falsepos/job_${SLURM_JOB_ID}
RESULTS=$REPO/reports/runs/fc_falsepos_results.txt
PARSE=$REPO/runs/harnesses/fc_falsepos_parse.py
PY=$REPO/.venv/bin/python
mkdir -p "$OUTROOT"
cd "$REPO"
COMMIT=$(git rev-parse HEAD)
CMD="python -m fcl_psp.models.run_model training.target_label=event_type model.model_name=mlp_classifier window_extraction.window_length=0.020 window_extraction.windows_local_dir=$WINDIR tracking.mode=disabled hydra.run.dir=$OUTROOT/hydra hydra.job.chdir=true"

echo "START fc_falsepos job=$SLURM_JOB_ID host=$(hostname) $(date)"
echo "COMMIT=$COMMIT"
echo "OUTROOT=$OUTROOT"
echo "CMD=$CMD"

# --- Step 1: train + persist per-window OOF predictions (y_pred + status) ---
"$PY" -m fcl_psp.models.run_model \
  training.target_label=event_type \
  model.model_name=mlp_classifier \
  window_extraction.window_length=0.020 \
  window_extraction.windows_local_dir="$WINDIR" \
  tracking.mode=disabled \
  hydra.run.dir="$OUTROOT/hydra" \
  hydra.job.chdir=true

# --- locate the OOF dump written by save_offline_outputs() ---
OOF=$(find "$OUTROOT" -name oof_predictions.parquet -print -quit)
echo "OOF_PARQUET=$OOF"
if [ -z "$OOF" ]; then
  echo "ERROR: oof_predictions.parquet not found under $OUTROOT" >&2
  exit 2
fi

# --- Step 2: parse -> naive FP + honest nuisance-trip rate (per-fold + pooled) ---
"$PY" "$PARSE" \
  --parquet "$OOF" \
  --out "$RESULTS" \
  --jobid "$SLURM_JOB_ID" \
  --commit "$COMMIT" \
  --cmd "$CMD"

echo "RESULTS=$RESULTS"
echo "DONE fc_falsepos job=$SLURM_JOB_ID $(date)"
