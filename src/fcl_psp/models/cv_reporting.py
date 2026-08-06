# fcl_psp/models/cv_reporting.py

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import wandb
from psp_helper.constants import TaskType
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_metrics_table_rows(
    rows: list[dict],
    keys: list[str],
) -> tuple[list[list[str]], list[float], list[float]]:
    table_data: list[list[str]] = []
    for i, metric in enumerate(rows, start=1):
        table_data.append([str(i)] + [f"{metric[k]:.4f}" for k in keys])

    means = [float(np.mean([m[k] for m in rows])) for k in keys]
    stds = [float(np.std([m[k] for m in rows])) for k in keys]
    return table_data, means, stds


def print_fold_metrics_tables(
    *,
    fold_metrics_all: list[dict],
    oof_parts: list[pd.DataFrame],
    task_type: str,
    logger: logging.Logger,
    fault_start_col: str = "status",
    fault_start_value: str = "fault_start",
) -> None:
    """
    Prints two tables:
      1) All windows (from fold_metrics_all)
      2) fault_start-only (computed from oof_parts)
    """
    if not fold_metrics_all:
        logger.error("No fold metrics to display.")
        return
    if not oof_parts:
        logger.error("No OOF parts available to compute fault_start-only metrics.")
        return

    # ---- Table 1: all windows ----
    keys = list(fold_metrics_all[0].keys())

    table_all, mean_all, std_all = _compute_metrics_table_rows(fold_metrics_all, keys)
    table_all += [["—"] + ["—" for _ in keys]]
    table_all += [["Mean"] + [f"{m:.3f}" for m in mean_all]]
    table_all += [["Std"] + [f"{s:.3f}" for s in std_all]]

    headers = ["Fold"] + keys
    logger.info(
        "\n"
        + tabulate(
            table_all, headers=headers, tablefmt="grid", stralign="center", numalign="center"
        )
    )

    # ---- Table 2: fault_start-only ----
    # Build per-fold metrics from each oof_df part
    fold_metrics_fs: list[dict] = []

    for fold_idx, oof_df in enumerate(oof_parts, start=1):
        if fault_start_col not in oof_df.columns:
            logger.warning(
                f"Cannot compute fault_start-only metrics: missing column '{fault_start_col}' in OOF data."
            )
            return

        mask = (oof_df[fault_start_col].astype(str) == fault_start_value).to_numpy()
        if not mask.any():
            # keep row but NaNs -> visible in table
            fold_metrics_fs.append({k: float("nan") for k in keys})
            continue

        y_true = oof_df.loc[mask, "y_true"].to_numpy()
        y_pred = oof_df.loc[mask, "y_pred"].to_numpy()

        if task_type in ("binary", "multiclass"):
            p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
            r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            fold_metrics_fs.append({"precision": p, "recall": r, "f1_score": f1})
        else:
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(root_mean_squared_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            fold_metrics_fs.append({"mae": mae, "rmse": rmse, "r2": r2})

    # keys might differ between cls/reg; recompute
    keys_fs = list(fold_metrics_fs[0].keys())
    table_fs, mean_fs, std_fs = _compute_metrics_table_rows(fold_metrics_fs, keys_fs)
    table_fs += [["—"] + ["—" for _ in keys_fs]]
    table_fs += [["Mean"] + [f"{m:.3f}" for m in mean_fs]]
    table_fs += [["Std"] + [f"{s:.3f}" for s in std_fs]]

    headers_fs = ["Fold"] + keys_fs
    logger.info(
        "\n"
        + "fault_start-only metrics:\n"
        + tabulate(
            table_fs, headers=headers_fs, tablefmt="grid", stralign="center", numalign="center"
        )
    )


def log_dataset_overview(
    *,
    window_data: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    task_type: str,
    logger: logging.Logger,
    fault_id_to_label: Dict[int, str] | None = None,
    max_classes: int = 20,
) -> None:
    """
    Logs a compact, readable overview of dataset alignment and target distribution.
    - classification (binary/multiclass): class distribution
    - regression: numeric summary stats (no fake 'classes')
    Intended for DEBUG-level logging.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug(
        "Dataset overview | "
        f"X={window_data.shape}, y={y.shape} (dtype={getattr(y, 'dtype', type(y))}), "
        f"sample_ids={sample_ids.shape}"
    )
    logger.debug(f"Unique sample groups: {len(np.unique(sample_ids))}")

    # ----------------------------
    # Classification
    # ----------------------------
    if task_type in ("binary", "multiclass"):
        unique, counts = np.unique(y, return_counts=True)
        total = int(counts.sum())

        rows = []
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            cls_int = int(cls)
            label = (
                fault_id_to_label.get(cls_int, str(cls_int))
                if fault_id_to_label is not None
                else str(cls_int)
            )
            frac = 100.0 * cnt / total
            rows.append((cls_int, label, int(cnt), frac))

        rows.sort(key=lambda r: r[2], reverse=True)

        logger.debug("Class distribution (sorted by support):")
        logger.debug("  id | label                   | count    | share")
        logger.debug("  ---+-------------------------+----------+--------")

        for i, (cls_int, label, cnt, frac) in enumerate(rows):
            if i >= max_classes:
                logger.debug(f"  ... ({len(rows) - max_classes} more classes)")
                break
            logger.debug(f"  {cls_int:>3} | {label:<23} | {cnt:>8} | {frac:6.2f}%")

        # small sanity checks that are often useful
        if fault_id_to_label is not None and 0 in fault_id_to_label:
            share_no_fault = next((frac for (cid, _, _, frac) in rows if cid == 0), None)
            if share_no_fault is not None:
                logger.debug(f"Share no_fault (class 0): {share_no_fault:.2f}%")

        return

    # ----------------------------
    # Regression
    # ----------------------------
    if task_type == "regression":
        y_arr = np.asarray(y, dtype=float)

        n = int(y_arr.size)
        n_nan = int(np.isnan(y_arr).sum())
        n_inf = int(np.isinf(y_arr).sum())
        finite = y_arr[np.isfinite(y_arr)]

        logger.debug(
            f"Target summary (regression): n={n}, finite={finite.size}, nan={n_nan}, inf={n_inf}"
        )
        if finite.size == 0:
            logger.debug("Target summary (regression): no finite values to summarize.")
            return

        # robust percentiles (good for quick sanity)
        p = np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 99, 100])

        logger.debug(
            "Target distribution (regression): "
            f"min={p[0]:.3f}, p1={p[1]:.3f}, p5={p[2]:.3f}, "
            f"p25={p[3]:.3f}, median={p[4]:.3f}, p75={p[5]:.3f}, "
            f"p95={p[6]:.3f}, p99={p[7]:.3f}, max={p[8]:.3f}"
        )

        mean = float(np.mean(finite))
        std = float(np.std(finite))
        logger.debug(f"Target mean±std (regression): {mean:.3f} ± {std:.3f}")

        # optional: “near-zero” mass can reveal many non-fault / default labels leaking in
        near_zero = float(np.mean(np.abs(finite) < 1e-6)) * 100.0
        logger.debug(f"Target near-zero share (|y|<1e-6): {near_zero:.2f}%")

        return

    # ----------------------------
    # Unknown task_type
    # ----------------------------
    logger.debug(f"Target overview: unsupported task_type='{task_type}' (no distribution logged).")


def validate_cv_results(
    *,
    oof_parts_len: int,
    fold_metrics: Sequence[Dict[str, float]],
) -> None:
    if oof_parts_len <= 0:
        raise ValueError("No OOF parts collected.")
    if not fold_metrics:
        raise ValueError("No fold metrics were collected.")


def cv_metric_spec(task_type: TaskType) -> Tuple[List[str], str, Literal["max", "min"]]:
    """
    Returns:
      keys_to_log, key_for_best_fold, best_mode
    """
    if task_type in ("binary", "multiclass"):
        return (["precision", "recall", "f1_score"], "f1_score", "max")
    if task_type == "regression":
        return (["mae", "rmse", "r2"], "mae", "min")
    raise ValueError(f"Unknown task_type='{task_type}'")


def log_cv_summary(
    *,
    fold_metrics: Sequence[Dict[str, float]],
    keys: Sequence[str],
    prefix: str = "cv",
    logger: logging.Logger | None = None,
) -> Dict[str, float]:
    """
    Computes and logs mean/std over folds into W&B.
    Optionally logs a readable summary via logger.info.
    Returns the summary dict.
    """
    summary: Dict[str, float] = {}

    for k in keys:
        vals = np.array([float(m[k]) for m in fold_metrics], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        summary[f"{prefix}/mean_{k}"] = mean
        summary[f"{prefix}/std_{k}"] = std

    # ---- W&B logging ----
    wandb.log(summary)
    for k, v in summary.items():
        wandb.summary[k] = v

    # ---- Human-readable log ----
    if logger is not None:
        metrics_str = " | ".join(
            f"{k}: {summary[f'{prefix}/mean_{k}']:.4f} ± {summary[f'{prefix}/std_{k}']:.4f}"
            for k in keys
        )
        logger.info(f"CV summary ({prefix}): {metrics_str}")

    return summary


def select_best_fold(
    *,
    fold_metrics: Sequence[Dict[str, float]],
    best_key: str,
    best_mode: Literal["max", "min"],
) -> int:
    scores = np.array([float(m[best_key]) for m in fold_metrics], dtype=float)
    if best_mode == "max":
        return int(np.argmax(scores))
    if best_mode == "min":
        return int(np.argmin(scores))
    raise ValueError(f"Invalid best_mode='{best_mode}'")
