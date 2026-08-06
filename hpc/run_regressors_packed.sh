#!/bin/bash -l
#SBATCH --job-name=ml_reg_packed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=./hpc/hpc_logs/%x/%x-%A_%a-on-%N.out

module load python/3.12-conda
conda activate juoe_ml

TOPOLOGY="hv_double_line_90kv"
TARGET_LABEL="y_fault_location"

REPOSITORY_NAME="fault-classification-localization"
DATASET_DIR="/home/woody/iwi5/<account>/datasets"
PROJECT_DIR="/home/hpc/iwi5/<account>/Repositories/${REPOSITORY_NAME}"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# Robust TMPDIR
if [ -z "${TMPDIR:-}" ]; then
  TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
fi
JOB_TMP_DIR="${TMPDIR%/}/${SLURM_JOB_ID}"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"
mkdir -p "$WINDOWS_TMP_PATH"

# Topology config mapping (keep your mapping if you need multiple topologies)
TOPOLOGY_CONFIG="hv_double_line_90kv"

# ----------------------------
# Packed run list for this array task
# ----------------------------
RUNS=()
case "${SLURM_ARRAY_TASK_ID}" in
  1)
    RUNS+=("extra_trees_regressor|0.050")
    RUNS+=("decision_tree_regressor|0.050")
    RUNS+=("decision_tree_regressor|0.020")
    RUNS+=("k_neighbors_regressor|0.040")
    ;;
  2)
    RUNS+=("ada_boost_regressor|0.050")
    RUNS+=("extra_trees_regressor|0.040")
    RUNS+=("voting_regressor|0.030")
    RUNS+=("voting_regressor|0.020")
    ;;
  3)
    RUNS+=("stacking_regressor|0.050")
    RUNS+=("stacking_regressor|0.040")
    RUNS+=("decision_tree_regressor|0.030")
    RUNS+=("mlp_regressor|0.030")
    RUNS+=("stacking_regressor|0.010")
    RUNS+=("voting_regressor|0.010")
    ;;
  4)
    RUNS+=("ada_boost_regressor|0.040")
    RUNS+=("extra_trees_regressor|0.030")
    RUNS+=("decision_tree_regressor|0.040")
    RUNS+=("ada_boost_regressor|0.030")
    RUNS+=("extra_trees_regressor|0.010")
    RUNS+=("mlp_regressor|0.020")
    RUNS+=("ada_boost_regressor|0.010")
    RUNS+=("k_neighbors_regressor|0.050")
    ;;
  5)
    RUNS+=("stacking_regressor|0.030")
    RUNS+=("voting_regressor|0.050")
    RUNS+=("extra_trees_regressor|0.020")
    RUNS+=("mlp_regressor|0.050")
    RUNS+=("voting_regressor|0.040")
    RUNS+=("mlp_regressor|0.040")
    RUNS+=("stacking_regressor|0.020")
    RUNS+=("ada_boost_regressor|0.020")
    RUNS+=("hist_gradient_boosting_regressor|0.050")
    RUNS+=("decision_tree_regressor|0.010")
    RUNS+=("hist_gradient_boosting_regressor|0.040")
    RUNS+=("hist_gradient_boosting_regressor|0.030")
    RUNS+=("mlp_regressor|0.010")
    RUNS+=("k_neighbors_regressor|0.030")
    RUNS+=("hist_gradient_boosting_regressor|0.020")
    RUNS+=("k_neighbors_regressor|0.020")
    RUNS+=("k_neighbors_regressor|0.010")
    RUNS+=("hist_gradient_boosting_regressor|0.010")
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Running on $(hostname) | Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | Start: $(date)"
echo "Runs in this task: ${#RUNS[@]}"

# ----------------------------
# Copy only needed windows for this task
# ----------------------------
SRC_DIR="$DATASET_DIR/windows_tmp"
EXTENSIONS=(raw parquet json)

WINDOWS=()
for spec in "${RUNS[@]}"; do
  W_FLOAT="${spec##*|}"                # 0.050
  W_TAG="$(printf "%0.3f" "$W_FLOAT" | sed 's/\./p/')"   # 0.050 -> 0p050
  WINDOWS+=("$W_TAG")
done
WINDOWS=($(printf "%s\n" "${WINDOWS[@]}" | sort -u))

for W_TAG in "${WINDOWS[@]}"; do
  for EXT in "${EXTENSIONS[@]}"; do
    FILE_PATTERN="*${TOPOLOGY}_W${W_TAG}_*.${EXT}"
    if find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -print -quit | grep -q .; then
      find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -exec cp -t "$WINDOWS_TMP_PATH" {} +
    else
      echo "WARNING: No ${EXT} files found for W=${W_TAG}"
    fi
  done
done

# ----------------------------
# Run loop
# ----------------------------
cd "$PROJECT_DIR"

for spec in "${RUNS[@]}"; do
  MODEL_NAME="${spec%%|*}"
  W_FLOAT="${spec##*|}"

  echo "=================================================="
  echo "Model=${MODEL_NAME} | W=${W_FLOAT}"
  echo "=================================================="

  python src/fcl_psp/models/run_model.py --multirun \
    dataset.topology="$TOPOLOGY" \
    dataset="$TOPOLOGY_CONFIG" \
    model.model_name="$MODEL_NAME" \
    training.target_label="$TARGET_LABEL" \
    window_extraction.window_length="${W_FLOAT}" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
    || echo "Model training failed: $MODEL_NAME @ $W_FLOAT"
done

echo "All packed regression jobs completed at $(date)"
