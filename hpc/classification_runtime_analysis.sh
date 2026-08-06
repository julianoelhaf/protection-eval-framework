#!/bin/bash -l
#SBATCH --job-name=ml_classification_runtime_analysis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --output=./hpc/hpc_logs/%x/%x-%j-on-%N.out

# -----------------------------------------
# Setup & Logging
# -----------------------------------------
mkdir -p ./hpc/hpc_logs/$SLURM_JOB_NAME
echo "Running on $(hostname) | Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "TMPDIR: ${TMPDIR:-UNSET!}"

module load python/3.12-conda
conda activate juoe_ml

# -----------------------------------------
# Configuration (defaults; allow sbatch --export overrides)
# -----------------------------------------
TOPOLOGY="hv_double_line_90kv"
WINDOW_LENGTHS=(
  # "0p010"
  # "0p020"
  # "0p030"
  # "0p040"
  "0p050"
)
# Default: 0.005 seconds (5 ms)
STEP_LENGTHS=(
  # "0p050"
  # "0p020"
  # "0p010"
  "0p005"
)

TARGET_LABEL="event_type"

CLASSIFIER_MODELS=(
  # "ada_boost_classifier"
  # "extra_trees_classifier"
  "hist_gradient_boosting_classifier"
  # "k_neighbors_classifier"
  # "logistic_regression"
  "mlp_classifier"
  # "random_forest_classifier"
  # "ridge_classifier"
)

REPOSITORY_NAME="fault-classification-localization"
DATASET_DIR="/home/woody/iwi5/<account>/datasets"
PROJECT_DIR="/home/hpc/iwi5/<account>/Repositories/${REPOSITORY_NAME}"
MLRUNS_DIR="file://${PROJECT_DIR}/mlruns"

JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# -----------------------------------------
# Robust temp dir (job-scoped; avoids collisions)
# -----------------------------------------
if [ -z "${TMPDIR:-}" ]; then
  echo "WARNING: TMPDIR is not set. Falling back to /tmp/$USER/$SLURM_JOB_ID"
  TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
fi

JOB_TMP_DIR="${TMPDIR%/}/${SLURM_JOB_ID}"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"
mkdir -p "$WINDOWS_TMP_PATH"
echo "Copying window files to: $WINDOWS_TMP_PATH"
START_TIME=$(date +%s)

SRC_DIR="$DATASET_DIR/windows_tmp"
EXTENSIONS=(raw parquet json)

for WINDOW_LENGTH in "${WINDOW_LENGTHS[@]}"; do
  for STEP_LENGTH_SECONDS in "${STEP_LENGTHS[@]}"; do
    for EXT in "${EXTENSIONS[@]}"; do
      FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH}_S${STEP_LENGTH_SECONDS}*.${EXT}"
      if find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -print -quit | grep -q .; then
        find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -exec cp -t "$WINDOWS_TMP_PATH" {} +
        echo "Copied ${EXT} files for window length: $WINDOW_LENGTH and step length: $STEP_LENGTH_SECONDS"
      else
        echo "WARNING: No ${EXT} files found for window length: $WINDOW_LENGTH and step length: $STEP_LENGTH_SECONDS"
      fi
    done
  done
done


COPY_DURATION=$(( $(date +%s) - START_TIME ))
echo "Data copy completed in ${COPY_DURATION} seconds."


# -----------------------------------------
# Topology -> dataset config mapping
# -----------------------------------------
TOPOLOGY_CONFIG="default"
case "$TOPOLOGY" in
  test_hv_double_line_90kv | hv_double_line_90kv) TOPOLOGY_CONFIG="hv_double_line_90kv" ;;
  test_hv_double_line_110kv | hv_double_line_110kv) TOPOLOGY_CONFIG="hv_double_line_110kv" ;;
  test_hv_reference_110kv | hv_reference_110kv) TOPOLOGY_CONFIG="hv_reference_110kv" ;;
  test_mini_mv_hybrid_ohl_cable_meshed_22kv | mv_hybrid_ohl_cable_meshed_22kv) TOPOLOGY_CONFIG="mv_hybrid_ohl_cable_meshed_22kv" ;;
  test_mv_cigre_distribution_20kv | mv_cigre_distribution_20kv) TOPOLOGY_CONFIG="mv_cigre_distribution_20kv" ;;
  test_ehv_ieee39_transmission_345kv | ehv_ieee39_transmission_345kv) TOPOLOGY_CONFIG="ehv_ieee39_transmission_345kv" ;;
  *) echo "WARNING: Unknown topology '$TOPOLOGY' → using default config" ;;
esac
echo "Using topology config: $TOPOLOGY_CONFIG"

# -----------------------------------------
# Run Classifiers
# -----------------------------------------
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd into PROJECT_DIR='$PROJECT_DIR'"; exit 1; }

for W in "${WINDOW_LENGTHS[@]}"; do
  for MODEL_NAME in "${CLASSIFIER_MODELS[@]}"; do
    for step_length_seconds in "${STEP_LENGTHS[@]}"; do

      W_FLOAT="${W/p/.}"   # 0p020 -> 0.020
      S_FLOAT="${step_length_seconds/p/.}" # 0p005 -> 0.005
      echo "==================================================================================================================================="
      echo "Window length: ${W} s (Hydra: ${W_FLOAT}), Model: ${MODEL_NAME}", "Step length: ${step_length_seconds} s (Hydra: ${S_FLOAT})"
      echo "==================================================================================================================================="
      
      python src/fcl_psp/models/run_model_runtime.py --multirun \
        dataset.topology="$TOPOLOGY" \
        dataset="$TOPOLOGY_CONFIG" \
        model.model_name="$MODEL_NAME" \
        training.target_label="$TARGET_LABEL" \
        window_extraction.window_length="${W_FLOAT}" \
        window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
        window_extraction.step_length_seconds="${S_FLOAT}" \
        || echo "Model training failed: $MODEL_NAME @ $W"

      echo "Completed: $MODEL_NAME for window length $W and step length $step_length_seconds"
    done
  done
done

echo "All classification jobs completed at $(date)"
