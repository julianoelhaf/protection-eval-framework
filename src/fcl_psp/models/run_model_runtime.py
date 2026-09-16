import copy
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import hydra
import numpy as np
import pandas as pd
import sklearn
import wandb
from psp_helper.config import MainConfig
from psp_helper.constants import FAULT_ID_TO_LABEL, FAULT_LABEL_TO_ID
from psp_helper.windows_helper import load_windows_and_labels
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fcl_psp.models import data_sparsity
from fcl_psp.models.cv_reporting import (
    cv_metric_spec,
    log_cv_summary,
    log_dataset_overview,
    print_fold_metrics_tables,
    select_best_fold,
    validate_cv_results,
)
from fcl_psp.models.model_utils import create_model_from_name, get_task_type
from fcl_psp.models.posthoc_analysis import get_run_out_dir, save_offline_outputs
from fcl_psp.models.posthoc_runner import run_posthoc_from_oof
from fcl_psp.models.wand_utils import wandb_log_dataset_params, wandb_log_model_params

# ---- Pick which label columns you want to carry into OOF ----
OOF_LABEL_COLS = [
    "sample_id",
    "event_type",
    "fault_class",
    "status",
    "y_fault_line",
    "y_fault_location",
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float, float]:
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    return float(precision), float(recall), float(f1)


def evaluate_regression_model(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float, float]:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}. R2 Score: {r2:.3f}")
    return float(mae), float(rmse), float(r2)


def build_fault_label(
    event_type: str,
    a: bool,
    b: bool,
    c: bool,
    is_grounded: bool,
    status: str,
) -> str:
    # Any window not containing the fault start is treated as no_fault
    if status != "fault_start":
        return "no_fault"

    phases = ("A" if a else "") + ("B" if b else "") + ("C" if c else "")
    if not phases:
        raise ValueError(
            "status='fault_start' but no phase selected: "
            f"event_type={event_type}, A={a}, B={b}, C={c}"
        )

    ground = "G" if is_grounded else ""
    return f"{event_type}_{phases}{ground}"


# --- add helper near the top of the file ---
def measure_predict_runtime(
    model,
    X_ref: np.ndarray,
    *,
    n_repeats: int = 30,
    n_warmup: int = 5,
    max_batch_samples: int = 2048,
) -> dict[str, float]:
    """Measure relative CPU-side inference cost for sklearn models.

    Reports:
      - single-sample latency
      - batch latency
      - batch per-sample latency
      - throughput
    """
    if len(X_ref) == 0:
        return {
            "predict_single_mean_s": np.nan,
            "predict_single_std_s": np.nan,
            "predict_batch_mean_s": np.nan,
            "predict_batch_std_s": np.nan,
            "predict_batch_per_sample_mean_s": np.nan,
            "predict_batch_per_sample_std_s": np.nan,
            "predict_throughput_samples_per_s": np.nan,
            "predict_batch_n_samples": 0,
        }

    X_single = X_ref[:1]
    X_batch = X_ref[: min(len(X_ref), max_batch_samples)]

    # Warm-up
    for _ in range(n_warmup):
        _ = model.predict(X_single)
        _ = model.predict(X_batch)

    single_times = []
    for _ in range(n_repeats):
        t0 = perf_counter()
        _ = model.predict(X_single)
        single_times.append(perf_counter() - t0)

    batch_times = []
    for _ in range(n_repeats):
        t0 = perf_counter()
        _ = model.predict(X_batch)
        batch_times.append(perf_counter() - t0)

    batch_per_sample = [t / len(X_batch) for t in batch_times]

    return {
        "predict_single_mean_s": float(np.mean(single_times)),
        "predict_single_std_s": (
            float(np.std(single_times, ddof=1)) if len(single_times) > 1 else 0.0
        ),
        "predict_batch_mean_s": float(np.mean(batch_times)),
        "predict_batch_std_s": float(np.std(batch_times, ddof=1)) if len(batch_times) > 1 else 0.0,
        "predict_batch_per_sample_mean_s": float(np.mean(batch_per_sample)),
        "predict_batch_per_sample_std_s": (
            float(np.std(batch_per_sample, ddof=1)) if len(batch_per_sample) > 1 else 0.0
        ),
        "predict_throughput_samples_per_s": float(len(X_batch) / np.mean(batch_times)),
        "predict_batch_n_samples": int(len(X_batch)),
    }


def create_fault_classes(labels: pd.DataFrame) -> List[int]:
    required_cols = {
        "event_type",
        "y_phase_A",
        "y_phase_B",
        "y_phase_C",
        "y_is_grounded",
        "status",
    }
    missing = required_cols - set(labels.columns)
    if missing:
        raise ValueError(
            f"Missing columns required for 'fault_class': {sorted(missing)}. "
            f"Found columns: {labels.columns.tolist()}"
        )

    cols = [
        "event_type",
        "y_phase_A",
        "y_phase_B",
        "y_phase_C",
        "y_is_grounded",
        "status",
    ]

    fault_classes: List[int] = []
    for event_type, a, b, c, grounded, status in labels[cols].itertuples(index=False, name=None):
        fault_label = build_fault_label(
            event_type=event_type,
            a=bool(a),
            b=bool(b),
            c=bool(c),
            is_grounded=bool(grounded),
            status=status,
        )

        try:
            fault_classes.append(FAULT_LABEL_TO_ID[fault_label])

        except KeyError as e:
            raise ValueError(
                f"Unknown fault label '{fault_label}'. " "Check FAULT_LABEL_TO_ID consistency."
            ) from e

    return fault_classes


def get_sample_ids_and_fault_targets(
    labels: pd.DataFrame, config: MainConfig
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if "sample_id" not in labels.columns:
        raise ValueError("Missing required column: 'sample_id' in labels DataFrame.")

    sample_ids = labels["sample_id"].to_numpy(dtype=int)
    target_label = config.training.target_label

    # Always create fault_class for downstream analysis (cheap + useful)
    labels["fault_class"] = create_fault_classes(labels)

    # ---- IMPORTANT: if target_label == 'event_type', you actually want fault_class (int IDs) ----
    if target_label == "event_type" or target_label == "fault_class":
        logger.debug(
            f"Target label '{target_label}' requested; using 'fault_class' as target variable."
        )
        y = labels["fault_class"].to_numpy(dtype=int)
        return sample_ids, y, labels

    if target_label not in labels.columns:
        raise ValueError(
            f"Fault target column '{target_label}' not found in labels DataFrame. "
            f"Available columns: {labels.columns.tolist()}"
        )

    # ---- Otherwise, use the requested label column as-is, with correct dtype ----
    fault_target_types = {
        "y_fault_present": int,
        "y_fault_location": float,
        # event_type handled above
    }

    if target_label not in fault_target_types:
        raise ValueError(
            f"Invalid target_label '{target_label}'. Supported labels: {list(fault_target_types.keys()) + ['event_type']}"
        )

    y = labels[target_label].to_numpy().astype(fault_target_types[target_label])
    logger.debug(f"Extracted {len(sample_ids)} samples with fault target '{target_label}'.")
    logger.debug(f"Shape of target variable y: {y.shape}")

    return sample_ids, y, labels


def select_relay_features(
    windows: np.ndarray,
    config: MainConfig,
) -> tuple[np.ndarray, dict]:
    """
    Select relay/channel subsets from windows of shape (N, L, F).

    Modes:
        - full
        - single_relay
        - relay_subset
        - drop_one_relay
    """
    if not hasattr(config, "ablation") or not config.ablation.enabled:
        return windows, {
            "enabled": False,
            "mode": "full",
            "selected_relays": list(range(8)),
            "n_features_out": int(windows.shape[2]),
        }

    mode = config.ablation.mode
    n_relays = int(config.ablation.n_relays)
    features_per_relay = int(config.ablation.features_per_relay)
    # total_expected = n_relays * features_per_relay

    # if windows.ndim != 3:
    #     raise ValueError(f"Expected windows with shape (N, L, F), got {windows.shape}")

    # if windows.shape[2] != total_expected:
    #     raise ValueError(
    #         f"Expected {total_expected} features "
    #         f"({n_relays} relays x {features_per_relay} features), "
    #         f"but got {windows.shape[2]}."
    #     )

    def relay_to_cols(relay_idx: int) -> list[int]:
        start = relay_idx * features_per_relay
        end = start + features_per_relay
        return list(range(start, end))

    all_relays = list(range(n_relays))

    if mode == "full":
        selected_relays = all_relays

    elif mode == "single_relay":
        relay_index = config.ablation.relay_index
        if relay_index is None:
            raise ValueError("ablation.relay_index must be set for mode='single_relay'")
        if relay_index not in all_relays:
            raise ValueError(f"relay_index must be in {all_relays}, got {relay_index}")
        selected_relays = [relay_index]

    elif mode == "relay_subset":
        relay_indices = list(config.ablation.relay_indices)
        if not relay_indices:
            raise ValueError("ablation.relay_indices must be non-empty for mode='relay_subset'")
        invalid = [r for r in relay_indices if r not in all_relays]
        if invalid:
            raise ValueError(f"Invalid relay_indices: {invalid}; valid range is {all_relays}")
        selected_relays = sorted(set(relay_indices))

    elif mode == "drop_one_relay":
        relay_index = config.ablation.relay_index
        if relay_index is None:
            raise ValueError("ablation.relay_index must be set for mode='drop_one_relay'")
        if relay_index not in all_relays:
            raise ValueError(f"relay_index must be in {all_relays}, got {relay_index}")
        selected_relays = [r for r in all_relays if r != relay_index]

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    selected_cols = []
    for relay_idx in selected_relays:
        selected_cols.extend(relay_to_cols(relay_idx))

    windows_selected = windows[:, :, selected_cols]

    meta = {
        "enabled": True,
        "mode": mode,
        "selected_relays": selected_relays,
        "selected_cols": selected_cols,
        "n_features_in": int(windows.shape[2]),
        "n_features_out": int(windows_selected.shape[2]),
    }

    if config.ablation.relay_index is not None:
        meta["relay_index"] = int(config.ablation.relay_index)

    return windows_selected, meta


def write_filtered_memmap(
    X: np.memmap,
    row_indices: np.ndarray,
    out_path: str | Path,
    dtype=np.float32,
    chunk_size: int = 4096,
) -> np.memmap:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_out = int(row_indices.size)
    L, F = int(X.shape[1]), int(X.shape[2])

    X_out = np.memmap(out_path, mode="w+", dtype=dtype, shape=(n_out, L, F))

    for i in range(0, n_out, chunk_size):
        j = min(i + chunk_size, n_out)
        idx = row_indices[i:j]
        X_out[i:j] = X[idx]  # bounded copy: only chunk_size rows at a time

    X_out.flush()
    return X_out


@hydra.main(version_base=None, config_path="../../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:
    start_time = time.time()
    logger.info("Starting training process...")

    # Load preprocessed windows and labels
    windows, labels, row_indices = load_windows_and_labels(config)

    # IMPORTANT: align windows with labels if fault-only filtering happened
    if row_indices is not None:
        # Make the filename deterministic per run configuration
        out_path = (
            Path(config.window_extraction.windows_local_dir)
            / f"X_fault_only_{config.dataset.topology}_W{str(config.window_extraction.window_length).replace('.','p')}.raw"
        )
        windows = write_filtered_memmap(windows, row_indices, out_path)
        logger.info(f"Filtered windows shape: {windows.shape}")  # should match labels rows now

    windows, ablation_meta = select_relay_features(windows, config)
    # logger.info(f"Ablation setup: {json.dumps(ablation_meta, indent=2)}")
    logger.info(f"Windows shape after relay selection: {windows.shape}")

    # Extract sample IDs and labels (adds fault_class if needed)
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, config)

    # Apply data sparsity transformation
    window_data, new_shape = data_sparsity.apply_sparsity_transform(windows, config)

    # Get task type from target_label
    task_type = get_task_type(config)

    # ----------------------------
    # W&B init
    # ----------------------------
    wandb.init(
        project=config.tracking.project,
        entity=config.tracking.entity,
        mode=config.tracking.mode,
    )

    # --- add after wandb.init(...) ---
    wandb.config.update(
        {
            "runtime_protocol": {
                "scope": "relative_cpu_side_model_runtime_only",
                "includes": [
                    "scaler.fit_transform",
                    "scaler.transform",
                    "model.fit",
                    "model.predict",
                ],
                "excludes": [
                    "disk_io",
                    "window_extraction",
                    "communication",
                    "synchronization",
                    "embedded_target_latency",
                ],
                "python_version": sys.version.split()[0],
                "sklearn_version": sklearn.__version__,
                "platform": platform.platform(),
                "cpu": platform.processor(),
            }
        },
        allow_val_change=True,
    )

    if task_type in ("binary", "multiclass"):
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                f"Classification target y must be integer class IDs, got dtype={y.dtype}. "
                "This would break FAULT_ID_TO_LABEL mapping."
            )

    wandb_log_dataset_params(
        n_samples=len(sample_ids), window_shape=new_shape, task_type=task_type, config=config
    )
    wandb_log_model_params(config)

    model = create_model_from_name(config)
    logger.info(f"Using model: {config.model.model_name}")

    grouped_k_fold = GroupKFold(n_splits=config.training.n_splits)

    fold_metrics = []
    fold_models = []
    oof_parts = []

    oof_label_cols = [c for c in OOF_LABEL_COLS if c in labels.columns]

    log_dataset_overview(
        window_data=window_data,
        y=y,
        sample_ids=sample_ids,
        task_type=task_type,
        logger=logger,
        fault_id_to_label=FAULT_ID_TO_LABEL,
    )

    # --- add before the CV loop ---
    fold_runtime = []

    for i, (train_idx, test_idx) in enumerate(
        grouped_k_fold.split(window_data, y, groups=sample_ids), start=1
    ):
        X_train, X_test = window_data[train_idx], window_data[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- replace the inside of the CV loop from scaler creation down to y_pred = ... with this ---
        fold_runtime_row: dict[str, float | int] = {
            "fold": i,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }

        scaler = StandardScaler(copy=False)

        t0 = perf_counter()
        X_train = scaler.fit_transform(X_train)
        fold_runtime_row["scaler_fit_transform_train_s"] = float(perf_counter() - t0)

        t0 = perf_counter()
        X_test = scaler.transform(X_test)
        fold_runtime_row["scaler_transform_test_s"] = perf_counter() - t0

        model_fold = copy.deepcopy(model)

        t0 = perf_counter()
        model_fold.fit(X_train, y_train)
        fold_runtime_row["model_fit_s"] = perf_counter() - t0
        logger.info(f"Split {i} training time: {fold_runtime_row['model_fit_s']:.2f} seconds")

        fold_models.append(model_fold)

        # ---- Predict + runtime ----
        predict_runtime = measure_predict_runtime(
            model_fold,
            X_test,
            n_repeats=30,
            n_warmup=5,
            max_batch_samples=2048,
        )
        fold_runtime_row.update(predict_runtime)

        t0 = perf_counter()
        y_pred = model_fold.predict(X_test)
        fold_runtime_row["predict_full_test_once_s"] = perf_counter() - t0

        fold_runtime.append(fold_runtime_row)

        logger.info(
            "Split %d runtime | fit=%.3fs | predict(single)=%.6fs | predict(batch/sample)=%.6fs | throughput=%.1f samples/s",
            i,
            fold_runtime_row["model_fit_s"],
            fold_runtime_row["predict_single_mean_s"],
            fold_runtime_row["predict_batch_per_sample_mean_s"],
            fold_runtime_row["predict_throughput_samples_per_s"],
        )

        # ---- Predict ----
        y_pred = model_fold.predict(X_test)

        # ---- Fold-level metrics (as before) ----
        metrics = (
            evaluate_classification_model(model_fold, X_test, y_test)
            if task_type in ("binary", "multiclass")
            else evaluate_regression_model(model_fold, X_test, y_test)
        )
        metric_names = (
            ["precision", "recall", "f1_score"]
            if task_type in ("binary", "multiclass")
            else ["mae", "rmse", "r2"]
        )
        fold_metrics.append(dict(zip(metric_names, metrics)))

        # ---- OOF rows: start from label slice (aligned by row) ----
        # Use .iloc because test_idx are positional indices
        # Sanity-check: only keep cols that exist

        oof_df = labels.iloc[test_idx][oof_label_cols].copy()
        oof_df["fold"] = i
        oof_df["y_true"] = y_test
        oof_df["y_pred"] = y_pred

        oof_parts.append(oof_df)

    # --- add after oof_df is created (after the folds loop) ---
    runtime_df = pd.DataFrame(fold_runtime)

    for col in [
        "scaler_fit_transform_train_s",
        "scaler_transform_test_s",
        "model_fit_s",
        "predict_single_mean_s",
        "predict_single_std_s",
        "predict_batch_mean_s",
        "predict_batch_std_s",
        "predict_batch_per_sample_mean_s",
        "predict_batch_per_sample_std_s",
        "predict_throughput_samples_per_s",
        "predict_full_test_once_s",
    ]:
        if col in runtime_df.columns:
            wandb.summary[f"runtime/{col}/mean"] = float(runtime_df[col].mean())
            wandb.summary[f"runtime/{col}/std"] = (
                float(runtime_df[col].std(ddof=1)) if len(runtime_df) > 1 else 0.0
            )

    logger.info(
        "Runtime summary | fit=%.3f±%.3fs | predict(single)=%.6f±%.6fs | predict(batch/sample)=%.6f±%.6fs | throughput=%.1f±%.1f samples/s",
        runtime_df["model_fit_s"].mean(),
        runtime_df["model_fit_s"].std(ddof=1) if len(runtime_df) > 1 else 0.0,
        runtime_df["predict_single_mean_s"].mean(),
        runtime_df["predict_single_mean_s"].std(ddof=1) if len(runtime_df) > 1 else 0.0,
        runtime_df["predict_batch_per_sample_mean_s"].mean(),
        runtime_df["predict_batch_per_sample_mean_s"].std(ddof=1) if len(runtime_df) > 1 else 0.0,
        runtime_df["predict_throughput_samples_per_s"].mean(),
        runtime_df["predict_throughput_samples_per_s"].std(ddof=1) if len(runtime_df) > 1 else 0.0,
    )

    # ---- after folds loop ----
    logger.info("All folds completed.")

    # Safety checks (no wandb.finish here; just raise)
    try:
        validate_cv_results(oof_parts_len=len(oof_parts), fold_metrics=fold_metrics)
    except ValueError as e:
        logger.error(str(e))
        wandb.summary["completed"] = False
        wandb.finish(exit_code=1)
        return

    oof_df = pd.concat(oof_parts, axis=0, ignore_index=True)

    out_dir = get_run_out_dir(config)

    # Save only columns you’ll want offline (keep it lean, but include your OOF label cols)
    keep_cols = [c for c in oof_label_cols if c in oof_df.columns] + ["fold", "y_true", "y_pred"]
    oof_df_to_save = oof_df[keep_cols].copy()

    # --- add before save_offline_outputs(...) ---
    runtime_df.to_csv(out_dir / "runtime_per_fold.csv", index=False)
    wandb.save(str(out_dir / "runtime_per_fold.csv"))

    save_offline_outputs(
        out_dir=out_dir,
        oof_df=oof_df_to_save,
        fold_metrics=fold_metrics,
        config=config,
    )

    # Posthoc is separate & optional
    run_posthoc_from_oof(
        oof_df=oof_df,
        task_type=task_type,
        prefix="cv/oof",
        enforce_all_known_classes=True,
        log_selection_metrics=True,
    )
    logger.info("Posthoc analysis completed.")

    print_fold_metrics_tables(
        fold_metrics_all=fold_metrics,
        oof_parts=oof_parts,
        task_type=task_type,
        logger=logger,
    )
    # ---- CV summary logging ----

    keys, best_key, best_mode = cv_metric_spec(task_type)
    log_cv_summary(fold_metrics=fold_metrics, keys=keys, prefix="cv")

    best_fold = select_best_fold(fold_metrics=fold_metrics, best_key=best_key, best_mode=best_mode)
    logger.info(f"Best fold index ({best_key}): {best_fold}")

    wandb.summary["completed"] = True
    wandb.summary["best_fold_index"] = best_fold

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training process completed in {elapsed_time:.2f} seconds.")

    wandb.finish()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    main()
