import glob
import logging
import os
import pickle
import time
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend (must precede pyplot import)
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# -----------------------------
# Config
# -----------------------------
N = 5
NUM_RUNS = 1000

# (note: duplicate keys will be overwritten in Python dicts; keep uniques)
model_run_ids = {
    "linear_svr": "4a8f6b83292541118c8f5a5589e8afb1",
    "ridge": "ee913c78c7fd4d6ca906fe63c73d9da0",
    "sgd_regressor": "1d3d8256a27e401fae9e541926cfe0ed",
    "stacking_regressor": "0afd55a4c63841aea14d7ce23142cfbf",
    "voting_regressor": "eaf1ba2d6d354a70abbd79b95cc03181",
    "k_neighbors_regressor": "071ff9ea4b074be3b53b708d8fdf30dd",
    "hist_gradient_boosting_regressor": "59646fde716640398d4b11c334e92011",
    "mlp_regressor": "f595f785a8124c5baa78704e33971823",
    "ada_boost_regressor": "1143b42281e84215abbc47001d61b05a",
    "decision_tree_regressor": "f3f19c676dd64f07bb5c84774a2338de",
    "extra_trees_regressor": "a412dd51077d433a85f992694dbdf36a",
    "linear_regression": "bbb621b5040a4d2185ef19a640b0ca89",
}

# Paths are configurable via environment variables; defaults are relative to the
# working directory so the script runs out-of-the-box without editing source.
DATA_FOLDER = os.environ.get("RUNTIME_DATA_FOLDER", os.path.join("data", "windows_tmp"))
X_FILENAME = os.environ.get("RUNTIME_X_FILENAME", "X_windows.npz")
Y_FILENAME = os.environ.get("RUNTIME_Y_FILENAME", "y_windows.npz")

# Outputs
PLOTS_DIR = os.environ.get("RUNTIME_PLOTS_DIR", os.path.join("outputs", "run_time_plots"))
CSV_DIR = os.environ.get("RUNTIME_CSV_DIR", os.path.join("outputs", "run_time_results"))
CSV_PATH = os.path.join(CSV_DIR, "model_runtimes.csv")


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("runtime-bench")


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)


def load_data(n: int) -> np.ndarray:
    x_path = os.path.join(DATA_FOLDER, X_FILENAME)
    y_path = os.path.join(DATA_FOLDER, Y_FILENAME)
    try:
        X = np.load(x_path, allow_pickle=True)["X"]
        _ = np.load(y_path, allow_pickle=True)["y"]  # loaded for completeness
        assert isinstance(X, np.ndarray)
        X = X[:n]
        logger.info(f"Loaded X with shape {X.shape}. Using first {n} samples.")
        return X
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def standardize_flatten(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Flatten to (n, d) and fit scaler on the N samples."""
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    return X_flat, scaler


# MLflow run store; override with MLRUNS_DIR (defaults to ./mlruns).
base_dir = os.environ.get("MLRUNS_DIR", "mlruns")


def load_model(run_id: str, model_name: str) -> Optional[object]:

    run_dir = os.path.join(base_dir, run_id, "artifacts")

    # Search recursively for model.pkl
    matches = glob.glob(os.path.join(run_dir, "**", "model.pkl"), recursive=True)
    if matches:
        model_path = matches[0]  # take first match if multiple
        print(f"{model_name}: {model_path}")

        with open(model_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"{model_name}: no model.pkl found")


def time_predictions(
    model, X_flat: np.ndarray, scaler: StandardScaler, num_runs: int
) -> np.ndarray:
    """
    Measure per-sample prediction time (ms) across num_runs * N predictions.
    Uses StandardScaler -> transform per-sample to mimic real-time pipeline.
    """
    runtimes_ms = []
    for _ in range(num_runs):
        for i in range(X_flat.shape[0]):
            x1 = X_flat[i].reshape(1, -1)
            x1 = scaler.transform(x1)
            t0 = time.perf_counter_ns()
            _ = model.predict(x1)  # single-sample prediction
            t1 = time.perf_counter_ns()
            runtimes_ms.append((t1 - t0) / 1_000_000.0)
    return np.array(runtimes_ms)


def plot_boxplot(runtimes_ms: np.ndarray, model_name: str) -> None:
    plt.figure(figsize=(8, 6))
    # Boxplot expects a sequence of arrays; pass [runtimes_ms]
    plt.boxplot([runtimes_ms], showfliers=False)
    plt.title(f"{model_name} Runtime over {NUM_RUNS} Runs (N={N} samples)")
    plt.ylabel("Runtime (ms)")
    plt.xlabel("Predictions")
    plt.grid(True)
    out_path = os.path.join(PLOTS_DIR, f"{model_name}_runtime_boxplot.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dirs()

    # Load and prep data
    X = load_data(N)
    X_flat, scaler = standardize_flatten(X)

    results = []

    for model_name, run_id in model_run_ids.items():
        logger.info(f"=== Processing model: {model_name} (run_id={run_id}) ===")
        model = load_model(run_id, model_name)
        if model is None:
            logger.warning(f"Skipping {model_name}: model not found/loaded.")
            continue

        try:
            runtimes_ms = time_predictions(model, X_flat, scaler, NUM_RUNS)
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            continue

        mean_ms = float(np.mean(runtimes_ms))
        std_ms = float(np.std(runtimes_ms))
        logger.info(f"{model_name} -> Mean: {mean_ms:.6f} ms, Std: {std_ms:.6f} ms")

        results.append({"model": model_name, "mean_runtime_ms": mean_ms, "std_runtime_ms": std_ms})

        # Plot
        plot_boxplot(runtimes_ms, model_name)

    # Save CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"Saved results CSV: {CSV_PATH}")
    else:
        logger.warning("No results to save (no models processed successfully).")


if __name__ == "__main__":
    main()
