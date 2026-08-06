#!/bin/bash -l
#SBATCH --job-name=ml_cls_ablation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
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
# USER CONFIG (safe defaults; can be overridden by sbatch --export)
# -----------------------------------------
TOPOLOGY="${TOPOLOGY:-hv_double_line_90kv}"
FAULT_TARGET="${FAULT_TARGET:-y_fault_location}"

# WINDOW_LENGTHS may arrive as a string via sbatch --export (not a bash array).
# Accept: "0p020" or "0p020 0p030" or "0p020,0p030"
if [ -n "${WINDOW_LENGTHS:-}" ]; then
  _wl="${WINDOW_LENGTHS//,/ }"
  read -r -a WINDOW_LENGTHS <<< "$_wl"
else
  WINDOW_LENGTHS=("0p030")
fi

# -----------------------------------------
# Model / ablation defaults (manual-friendly)
# -----------------------------------------
MODEL_NAME="${MODEL_NAME:-mlp_regressor}"
ABLATION_PARAM="${ABLATION_PARAM:-baseline}"

# Optional: warn when defaults are used
if [ -z "${MODEL_NAME+x}" ] || [ -z "${ABLATION_PARAM+x}" ]; then
  echo "INFO: MODEL_NAME or ABLATION_PARAM not provided via sbatch --export"
  echo "      Using defaults: MODEL_NAME=$MODEL_NAME, ABLATION_PARAM=$ABLATION_PARAM"
fi

echo "Using model: $MODEL_NAME"
echo "Ablation param: $ABLATION_PARAM"


# -----------------------------------------
# Paths
# -----------------------------------------
REPOSITORY_NAME="fault-classification-localization"
DATASET_DIR="/home/woody/iwi5/<account>/datasets"
PROJECT_DIR="/home/hpc/iwi5/<account>/Repositories/${REPOSITORY_NAME}"


export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# -----------------------------------------
# Robust temp dir (no job cancellation)
# -----------------------------------------
if [ -z "${TMPDIR:-}" ]; then
  echo "WARNING: TMPDIR is not set. Falling back to /tmp/$USER/$SLURM_JOB_ID"
  TMPDIR="/tmp/$USER/$SLURM_JOB_ID"
fi

WINDOWS_TMP_PATH="$TMPDIR/windows_tmp"

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
# File Preparation (copy window files)
# -----------------------------------------
mkdir -p "$WINDOWS_TMP_PATH"
echo "Copying window files to: $WINDOWS_TMP_PATH"
START_TIME=$(date +%s)

SRC_DIR="$DATASET_DIR/windows_tmp"
EXTENSIONS=(raw parquet json)

for WINDOW_LENGTH in "${WINDOW_LENGTHS[@]}"; do
  for EXT in "${EXTENSIONS[@]}"; do
    FILE_PATTERN="*${TOPOLOGY}_W${WINDOW_LENGTH}_*.${EXT}"
    if find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -print -quit | grep -q .; then
      find "$SRC_DIR" -maxdepth 1 -type f -name "$FILE_PATTERN" -exec cp -t "$WINDOWS_TMP_PATH" {} +
      echo "Copied ${EXT} files for window length: $WINDOW_LENGTH"
    else
      echo "WARNING: No ${EXT} files found for window length: $WINDOW_LENGTH"
    fi
  done
done

COPY_DURATION=$(( $(date +%s) - START_TIME ))
echo "Data copy completed in ${COPY_DURATION} seconds."

# -----------------------------------------
# Run Ablations
# -----------------------------------------
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd into PROJECT_DIR='$PROJECT_DIR'"; exit 1; }

# -----------------------------------------
# Baselines = strict scikit-learn defaults
# (Ablation varies exactly one knob; everything else stays default)
# -----------------------------------------

# HistGradientBoosting* defaults (Classifier/Regressor share these defaults)
HGB_BASE=(
  "model.hgb.l2_regularization=0.0"    # default: 0.0
  "model.hgb.learning_rate=0.1"        # default: 0.1
  "model.hgb.max_depth=null"           # default: None
  "model.hgb.max_iter=100"             # default: 100
  "model.hgb.min_samples_leaf=20"      # default: 20
)

# MLP* defaults (Classifier/Regressor share these defaults)
# hidden_layer_sizes default: (100,) -> YAML list [100]
MLP_BASE=(
  "model.mlp.activation=relu"          # default: relu
  "model.mlp.alpha=1e-4"               # default: 0.0001
  "model.mlp.batch_size=auto"          # default: auto
  "model.mlp.early_stopping=false"     # default: False
  "model.mlp.hidden_layer_sizes=[100]" # default: (100,)
  "model.mlp.learning_rate_init=1e-3"  # default: 0.001
  "model.mlp.max_iter=200"             # default: 200
  "model.mlp.n_iter_no_change=10"      # default: 10
)

# Build the ablation grid (exactly one param varies)
ABLATION_GRID=()

# -----------------------------------------
# Ablation grid (single varying parameter)
# -----------------------------------------
case "$ABLATION_PARAM" in
  baseline)
    echo "Running baseline (no ablation) for MODEL_NAME='$MODEL_NAME'"
    ABLATION_GRID=()
    ;;
  hgb_l2_regularization)    ABLATION_GRID=("model.hgb.l2_regularization=1e-4,1e-2") ;;
  hgb_learning_rate)        ABLATION_GRID=("model.hgb.learning_rate=0.03,0.2") ;;
  hgb_max_depth)            ABLATION_GRID=("model.hgb.max_depth=3,5,10") ;;
  hgb_max_iter)             ABLATION_GRID=("model.hgb.max_iter=50,300") ;;
  hgb_min_samples_leaf)     ABLATION_GRID=("model.hgb.min_samples_leaf=5,50") ;;

  mlp_alpha)                ABLATION_GRID=("model.mlp.alpha=1e-5,1e-3") ;;
  mlp_batch_size)           ABLATION_GRID=("model.mlp.batch_size=64,128,256") ;;
  mlp_early_stopping)       ABLATION_GRID=("model.mlp.early_stopping=true") ;;
  mlp_hidden_layer_sizes) ABLATION_GRID=("model.mlp.hidden_layer_sizes=[50],[100,50],[256,128]") ;;
  mlp_learning_rate_init)   ABLATION_GRID=("model.mlp.learning_rate_init=1e-5,1e-4,1e-2") ;;
  mlp_max_iter)             ABLATION_GRID=("model.mlp.max_iter=100,300,400") ;;
  mlp_n_iter_no_change)     ABLATION_GRID=("model.mlp.n_iter_no_change=5,20") ;;
  *)
    echo "ERROR: Unknown ABLATION_PARAM='$ABLATION_PARAM'"
    exit 1
    ;;
esac

# Pick baseline args
BASE_ARGS=()
case "$MODEL_NAME" in
  hist_gradient_boosting_classifier|hist_gradient_boosting_regressor)
    BASE_ARGS=("${HGB_BASE[@]}")
    ;;
  mlp_classifier|mlp_regressor)
    BASE_ARGS=("${MLP_BASE[@]}")
    ;;
  *)
    echo "ERROR: Unsupported MODEL_NAME='$MODEL_NAME' (expected HGB/MLP classifier or regressor)"
    exit 1
    ;;
esac

HYDRA_FLAGS=()
[ "${#ABLATION_GRID[@]}" -gt 0 ] && HYDRA_FLAGS+=(--multirun)


for W in "${WINDOW_LENGTHS[@]}"; do
  W_FLOAT="${W/p/.}"

  echo "=========================================================================="
  echo "Window length: ${W} s (Hydra: ${W_FLOAT})" MODEL_NAME="${MODEL_NAME}" ABLATION_PARAM="${ABLATION_PARAM}"
  echo "=========================================================================="

  python src/fcl_psp/models/run_model.py \
    "${HYDRA_FLAGS[@]}" \
    dataset.topology="$TOPOLOGY" \
    dataset="$TOPOLOGY_CONFIG" \
    model.model_name="$MODEL_NAME" \
    training.target_label="$FAULT_TARGET" \
    window_extraction.window_length="${W_FLOAT}" \
    window_extraction.windows_local_dir="$WINDOWS_TMP_PATH" \
    "${BASE_ARGS[@]}" \
    "${ABLATION_GRID[@]}" \
    "hydra.sweep.dir=${PROJECT_DIR}/outputs/multirun/abl_${MODEL_NAME}_${ABLATION_PARAM}_${SLURM_JOB_ID}_W${W}" \
    || echo "Model training failed: ${MODEL_NAME} @ ${W} (ablation=${ABLATION_PARAM})"

  echo "Completed: ${MODEL_NAME} for window length ${W} (ablation=${ABLATION_PARAM})"
done

echo "All ablation jobs completed at $(date)"
# -----------------------------------------