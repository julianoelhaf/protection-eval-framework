#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# submit_all_ablations_reg.sh
# - submits one SLURM job per (model, ablation_param) for FL
#
# Usage:
#   DRY_RUN=1 bash hpc/submit_all_ablations_reg.sh
#   bash hpc/submit_all_ablations_reg.sh
#
# ------------------------------------------------------------

SLURM_SCRIPT="${SLURM_SCRIPT:-hpc/run_ablation_studies.sh}"
DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-}"
TIME="${TIME:-24:00:00}"

# FL defaults (override via env if you want)
TOPOLOGY="${TOPOLOGY:-hv_double_line_90kv}"
WINDOW_LENGTHS="${WINDOW_LENGTHS:-0p040}"
FAULT_TARGET="${FAULT_TARGET:-y_fault_location}"  # <-- regression label for localization

if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: SLURM_SCRIPT not found: $SLURM_SCRIPT"
  exit 1
fi

declare -a JOB_SPECS=(
  # ------------------------
  # Baselines (no ablation)
  # ------------------------
  # "hist_gradient_boosting_regressor baseline"
  # "mlp_regressor baseline"

  # ------------------------
  # HGB regressor ablations
  # ------------------------
  # "hist_gradient_boosting_regressor hgb_l2_regularization"
  # "hist_gradient_boosting_regressor hgb_learning_rate"
  # "hist_gradient_boosting_regressor hgb_max_depth"
  # "hist_gradient_boosting_regressor hgb_max_iter"
  # "hist_gradient_boosting_regressor hgb_min_samples_leaf"

  # ------------------------
  # MLP regressor ablations
  # ------------------------
  # "mlp_regressor mlp_hidden_layer_sizes"
  "mlp_regressor mlp_batch_size"
  "mlp_regressor mlp_learning_rate_init"
  "mlp_regressor mlp_max_iter"
  "mlp_regressor mlp_alpha"
)


sbatch_cmd_base=(sbatch)
if [ -n "$PARTITION" ]; then sbatch_cmd_base+=(--partition="$PARTITION"); fi
if [ -n "$TIME" ]; then sbatch_cmd_base+=(--time="$TIME"); fi

echo "Using SLURM script: $SLURM_SCRIPT"
echo "Mode: FL (regression)"
echo "TOPOLOGY=$TOPOLOGY | WINDOW_LENGTHS=$WINDOW_LENGTHS | FAULT_TARGET=$FAULT_TARGET"
echo "Jobs to submit: ${#JOB_SPECS[@]}"
echo

for spec in "${JOB_SPECS[@]}"; do
  read -r model_name ablation_param <<< "$spec"

  job_name="abl_fl_${model_name}_${ablation_param}"

  sbatch_cmd=("${sbatch_cmd_base[@]}"
    --job-name="$job_name"
    --export=ALL,TOPOLOGY="$TOPOLOGY",WINDOW_LENGTHS="$WINDOW_LENGTHS",FAULT_TARGET="$FAULT_TARGET",MODEL_NAME="$model_name",ABLATION_PARAM="$ablation_param"
    "$SLURM_SCRIPT"
  )

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] ${sbatch_cmd[*]}"
  else
    echo "Submitting: $job_name"
    "${sbatch_cmd[@]}"
  fi
done

echo
echo "Done."
