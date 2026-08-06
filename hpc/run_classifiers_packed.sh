#!/bin/bash -l
#SBATCH --job-name=ml_cls_packed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=./hpc/hpc_logs/%x/%x-%A_%a-on-%N.out

module load python/3.12-conda
conda activate juoe_ml

TOPOLOGY="hv_double_line_90kv"
FAULT_TARGET="event_type"

REPOSITORY_NAME="fault-classification-localization"
DATASET_DIR="/home/woody/iwi5/<account>/datasets"
PROJECT_DIR="/home/hpc/iwi5/<account>/Repositories/${REPOSITORY_NAME}"

# TMP
if [ -z "${TMPDIR:-}" ]; then
  TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
fi
JOB_TMP_DIR="$TMPDIR/$SLURM_JOB_ID"
WINDOWS_TMP_PATH="$JOB_TMP_DIR/windows_tmp"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# topology config mapping (keep yours)
TOPOLOGY_CONFIG="hv_double_line_90kv"

# ---- pick runs for this array task ----
RUNS=()
case "${SLURM_ARRAY_TASK_ID}" in
  1)
    RUNS+=("ada_boost_classifier|0.050")
    ;;
  2)
    RUNS+=("ada_boost_classifier|0.040")
    RUNS+=("random_forest_classifier|0.040")
    ;;
  3)
    RUNS+=("ada_boost_classifier|0.030")
    RUNS+=("ada_boost_classifier|0.010")
    ;;
  4)
    RUNS+=("ada_boost_classifier|0.020")
    RUNS+=("random_forest_classifier|0.010")
    RUNS+=("random_forest_classifier|0.020")
    RUNS+=("random_forest_classifier|0.030")
    RUNS+=("random_forest_classifier|0.050")
    RUNS+=("logistic_regression|0.040")
    RUNS+=("mlp_classifier|0.040")
    RUNS+=("hist_gradient_boosting_classifier|0.050")
    ;;
  5)
    # cheap bundle
    RUNS+=("extra_trees_classifier|0.010")
    RUNS+=("extra_trees_classifier|0.020")
    RUNS+=("extra_trees_classifier|0.030")
    RUNS+=("extra_trees_classifier|0.040")
    RUNS+=("extra_trees_classifier|0.050")

    RUNS+=("hist_gradient_boosting_classifier|0.010")
    RUNS+=("hist_gradient_boosting_classifier|0.020")
    RUNS+=("hist_gradient_boosting_classifier|0.030")
    RUNS+=("hist_gradient_boosting_classifier|0.040")

    RUNS+=("k_neighbors_classifier|0.010")
    RUNS+=("k_neighbors_classifier|0.020")
    RUNS+=("k_neighbors_classifier|0.030")
    RUNS+=("k_neighbors_classifier|0.040")
    RUNS+=("k_neighbors_classifier|0.050")

    RUNS+=("logistic_regression|0.010")
    RUNS+=("logistic_regression|0.020")
    RUNS+=("logistic_regression|0.030")
    RUNS+=("logistic_regression|0.050")

    RUNS+=("mlp_classifier|0.010")
    RUNS+=("mlp_classifier|0.020")
    RUNS+=("mlp_classifier|0.030")
    RUNS+=("mlp_classifier|0.050")

    RUNS+=("ridge_classifier|0.010")
    RUNS+=("ridge_classifier|0.020")
    RUNS+=("ridge_classifier|0.030")
    RUNS+=("ridge_classifier|0.040")
    RUNS+=("ridge_classifier|0.050")
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

echo "Running on $(hostname) | Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} | Start: $(date)"
echo "Runs in this task: ${#RUNS[@]}"

# ---- Copy only needed window files for this task (reduces IO + time) ----
mkdir -p "$WINDOWS_TMP_PATH"
SRC_DIR="$DATASET_DIR/windows_tmp"
EXTENSIONS=(raw parquet json)

# collect unique window lengths needed in this task
WINDOWS=()
for spec in "${RUNS[@]}"; do
  W_FLOAT="${spec##*|}"          # "0.050"
  W_TAG="$(printf "%0.3f" "$W_FLOAT" | sed 's/\./p/')"   # 0.050 -> 0p050
  WINDOWS+=("$W_TAG")
done
# unique
WINDOWS=($(printf "%s\n" "${WINDOWS[@]}" | sort -u))

echo "Copying windows for: ${WINDOWS[*]}"
for W_TAG in "${WINDOWS[@]}"; do
  for EXT in "${EXTENSIONS[@]}"; do
    FILE_PATTERN="*${TOPOLOGY}_W${W_TAG}_*.${EXT}"
    if find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -print -quit | grep -q .; then
      find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -exec cp -t "$WINDOWS_TMP_PATH" {} +
    else
      echo "WARNING: missing ${EXT} for W=${W_TAG}"
    fi
  done
done

# ---- Run loop ----
cd "$PROJECT_DIR"

for spec in "${RUNS[@]}"; do
  MODEL_NAME="${spec%%|*}"
  W_FLOAT="${spec##*|}"

  echo "=================================================="
  echo "Model=${MODEL_NAME} | W=${W_FLOAT}"
  echo "=================================================="

  python src/fcl_psp/models/run_model.py --multirun \
    model.model_name="$MODEL_NAME" \
    window_extraction.window_length="$W_FLOAT" \
    training.target_label="$FAULT_TARGET" \
    dataset="$TOPOLOGY_CONFIG" \
    dataset.topology="$TOPOLOGY" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
    || echo "Model training failed: ${MODEL_NAME} @ ${W_FLOAT}"
done

echo "Done at $(date)"
