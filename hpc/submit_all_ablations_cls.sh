#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# submit_all_ablations_cls.sh
# - submits one SLURM job per (model, ablation_param) for FC
#
# Usage:
#   DRY_RUN=1 bash hpc/submit_all_ablations_cls.sh
#   bash hpc/submit_all_ablations_cls.sh
#
# ------------------------------------------------------------

SLURM_SCRIPT="${SLURM_SCRIPT:-hpc/run_ablation_studies.sh}"
DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-}"
TIME="${TIME:-24:00:00}"

# FC defaults (override via env if you want)
TOPOLOGY="${TOPOLOGY:-hv_double_line_90kv}"
WINDOW_LENGTHS="${WINDOW_LENGTHS:-0p040}" 
FAULT_TARGET="${FAULT_TARGET:-event_type}"  # classification label

# normalize once (accept "0p010,0p020" or "0p010 0p020")
_wl="${WINDOW_LENGTHS//,/ }"
_wl="$(echo "$_wl" | xargs)"  # optional: trim whitespace


if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: SLURM_SCRIPT not found: $SLURM_SCRIPT"
  exit 1
fi

declare -a JOB_SPECS=(
  # ------------------------
  # Baselines (no ablation)
  # ------------------------
  # "hist_gradient_boosting_classifier baseline"
  # "mlp_classifier baseline"
  
  # ------------------------
  # HGB classifier ablations
  # ------------------------
  # "hist_gradient_boosting_classifier hgb_l2_regularization"
  # "hist_gradient_boosting_classifier hgb_learning_rate"
  # "hist_gradient_boosting_classifier hgb_max_depth"
  # "hist_gradient_boosting_classifier hgb_max_iter"
  # "hist_gradient_boosting_classifier hgb_min_samples_leaf"

  # ------------------------
  # MLP classifier ablations
  # ------------------------
  # "mlp_classifier mlp_alpha"
  # "mlp_classifier mlp_batch_size"
  "mlp_classifier mlp_hidden_layer_sizes"
  # "mlp_classifier mlp_learning_rate_init"
  # "mlp_classifier mlp_max_iter"
)

sbatch_cmd_base=(sbatch)
if [ -n "$PARTITION" ]; then sbatch_cmd_base+=(--partition="$PARTITION"); fi
if [ -n "$TIME" ]; then sbatch_cmd_base+=(--time="$TIME"); fi

echo "Using SLURM script: $SLURM_SCRIPT"
echo "Mode: FC (classification)"
echo "TOPOLOGY=$TOPOLOGY | WINDOW_LENGTHS=$WINDOW_LENGTHS | FAULT_TARGET=$FAULT_TARGET"
echo "Jobs to submit: ${#JOB_SPECS[@]}"
echo

for spec in "${JOB_SPECS[@]}"; do
  read -r model_name ablation_param <<< "$spec"
  job_name="abl_fc_${model_name}_${ablation_param}"

  exports="ALL,TOPOLOGY=${TOPOLOGY},WINDOW_LENGTHS=${_wl},FAULT_TARGET=${FAULT_TARGET},MODEL_NAME=${model_name},ABLATION_PARAM=${ablation_param}"

  sbatch_cmd=("${sbatch_cmd_base[@]}"
    --job-name="$job_name"
    --export="$exports"
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
