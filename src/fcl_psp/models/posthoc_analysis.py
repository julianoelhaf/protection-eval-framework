# posthoc_analysis.py
# Refactor: uses FAULT_ID_TO_LABEL to attach human-readable labels for analysis + confusion matrices.

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb
from psp_helper.config import MainConfig
from psp_helper.constants import FAULT_ID_TO_LABEL
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}. Found: {df.columns.tolist()}"
        )


def _safe_int(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _as_wandb_table(rows: List[Dict[str, Any]]) -> "wandb.Table":
    if not rows:
        return wandb.Table(columns=["empty"], data=[["no rows"]])
    cols = sorted({k for r in rows for k in r.keys()})
    data = [[r.get(c, None) for c in cols] for r in rows]
    return wandb.Table(columns=cols, data=data)


def _infer_present_class_ids(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # stable numeric ordering
    ids = np.unique(np.concatenate([y_true, y_pred]).astype(int))
    return np.sort(ids)


def _class_id_to_label(cid: int) -> str:
    # fall back if mapping incomplete
    return str(FAULT_ID_TO_LABEL.get(int(cid), f"class_{int(cid)}"))


def add_fault_labels_columns(
    oof: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    out_true_label_col: str = "y_true_label",
    out_pred_label_col: str = "y_pred_label",
) -> pd.DataFrame:
    """
    Adds readable label columns (based on FAULT_ID_TO_LABEL) to an OOF dataframe.
    Does NOT modify input df; returns a copy.
    """
    _require_cols(oof, [y_true_col, y_pred_col], "add_fault_labels_columns")
    df = oof.copy()

    # Ensure ints for mapping
    df[y_true_col] = df[y_true_col].astype(int)
    df[y_pred_col] = df[y_pred_col].astype(int)

    df[out_true_label_col] = df[y_true_col].map(_class_id_to_label).astype(str)
    df[out_pred_label_col] = df[y_pred_col].map(_class_id_to_label).astype(str)
    return df


def _resolve_class_ids_and_names(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    enforce_all_known: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      - class_ids: numeric ids to use as `labels=` for confusion matrix / metrics
      - class_names: aligned readable labels via FAULT_ID_TO_LABEL

    By default it uses only classes present in (y_true ∪ y_pred) for compact plots/tables.
    Set enforce_all_known=True to include all ids in FAULT_ID_TO_LABEL (stable across runs).
    """
    if enforce_all_known:
        class_ids = np.array(sorted(int(k) for k in FAULT_ID_TO_LABEL.keys()), dtype=int)
    else:
        class_ids = _infer_present_class_ids(y_true, y_pred)

    class_names = [_class_id_to_label(int(i)) for i in class_ids]
    return class_ids, class_names


def _log_confusion_matrix_with_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_ids: np.ndarray,
    class_names: List[str],
    prefix: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    # Normalize by true row sums
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    # absolute
    abs_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(cm):
        r: Dict[str, Any] = {"true": class_names[i]}
        for j, v in enumerate(row):
            r[class_names[j]] = _safe_int(v)
        abs_rows.append(r)
    wandb.log({f"{prefix}/confusion_matrix": _as_wandb_table(abs_rows)})

    # normalized
    norm_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(cm_norm):
        r = {"true": class_names[i]}
        for j, v in enumerate(row):
            r[class_names[j]] = float(v)
        norm_rows.append(r)
    wandb.log({f"{prefix}/confusion_matrix_normalized": _as_wandb_table(norm_rows)})


def _per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_ids: np.ndarray,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, zero_division=0, average=None
    )
    rows: List[Dict[str, Any]] = []
    for name, pi, ri, fi, si in zip(class_names, p, r, f1, s):  # type: ignore
        rows.append(
            {
                "fault_label": name,
                "precision": float(pi),
                "recall": float(ri),
                "f1": float(fi),
                "support": int(si),
            }
        )
    # Sort by support (descending) to surface dominant classes
    rows.sort(key=lambda d: d["support"], reverse=True)
    return rows


# =============================================================================
# 4) Classification post-hoc (uses FAULT_ID_TO_LABEL)
# =============================================================================


def classification_posthoc_analysis(
    oof: pd.DataFrame,
    *,
    prefix: str = "cv/oof",
    fault_line_col: str = "y_fault_line",
    fault_start_col: str = "status",
    fault_start_value: str = "fault_start",
    min_support_line: int = 30,
    log_confusion: bool = True,
    log_fault_start_only: bool = True,
    enforce_all_known_classes: bool = False,
    log_fault_only: bool = True,
    no_fault_class_id: int = 0,
    log_all_windows_fault_line: bool = True,
) -> Dict[str, Any]:
    """
    Required columns: y_true, y_pred
    Recommended: status, y_fault_line

    Logs (if available / enabled):
      - Global per-class metrics + confusion matrix (all windows)
      - fault_start_only per-class metrics + confusion matrix
      - Per-fault-line macro/weighted F1 for explicit bases:
          * all_windows
          * fault_start_only
          * fault_only
          * fault_only_fault_start
    """
    _require_cols(oof, ["y_true", "y_pred"], "classification_posthoc_analysis")

    # Ensure ints for stable mapping
    y_true = oof["y_true"].to_numpy().astype(int)
    y_pred = oof["y_pred"].to_numpy().astype(int)

    class_ids, class_names = _resolve_class_ids_and_names(
        y_true, y_pred, enforce_all_known=enforce_all_known_classes
    )

    # -------------------------------------------------------------------------
    # 1) Global report + per-class table + confusion
    # -------------------------------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    wandb.log({f"{prefix}/classification_report": report})

    rows_global = _per_class_table(y_true, y_pred, class_ids=class_ids, class_names=class_names)
    wandb.log({f"{prefix}/per_class_metrics": _as_wandb_table(rows_global)})

    if log_confusion:
        _log_confusion_matrix_with_labels(
            y_true=y_true,
            y_pred=y_pred,
            class_ids=class_ids,
            class_names=class_names,
            prefix=prefix,
        )

    results: Dict[str, Any] = {
        "global_report": report,
        "global_per_class_rows": rows_global,
    }

    # -------------------------------------------------------------------------
    # 2) fault_start_only subset (explicit)
    # -------------------------------------------------------------------------
    if log_fault_start_only and (fault_start_col in oof.columns):
        mask_fs = oof[fault_start_col].astype(str).to_numpy() == fault_start_value
        if mask_fs.any():
            yt_fs = y_true[mask_fs]
            yp_fs = y_pred[mask_fs]

            class_ids_fs, class_names_fs = _resolve_class_ids_and_names(
                yt_fs, yp_fs, enforce_all_known=enforce_all_known_classes
            )

            report_fs = classification_report(
                yt_fs,
                yp_fs,
                labels=class_ids_fs,
                target_names=class_names_fs,
                output_dict=True,
                zero_division=0,
            )
            wandb.log({f"{prefix}/fault_start_only/classification_report": report_fs})

            rows_fs = _per_class_table(
                yt_fs, yp_fs, class_ids=class_ids_fs, class_names=class_names_fs
            )
            wandb.log({f"{prefix}/fault_start_only/per_class_metrics": _as_wandb_table(rows_fs)})

            if log_confusion:
                _log_confusion_matrix_with_labels(
                    y_true=yt_fs,
                    y_pred=yp_fs,
                    class_ids=class_ids_fs,
                    class_names=class_names_fs,
                    prefix=f"{prefix}/fault_start_only",
                )

            results["fault_start_only_report"] = report_fs
            results["fault_start_only_per_class_rows"] = rows_fs

    # -------------------------------------------------------------------------
    # 3) Per-fault-line analysis (explicit bases)
    # -------------------------------------------------------------------------
    def _per_fault_line_f1(df: pd.DataFrame, *, base_name: str) -> List[Dict[str, Any]]:
        if fault_line_col not in df.columns or len(df) == 0:
            return []

        rows: List[Dict[str, Any]] = []
        for line, g in df.groupby(fault_line_col, dropna=False, observed=False):
            n = len(g)
            if n < min_support_line:
                continue

            yt = g["y_true"].to_numpy().astype(int)
            yp = g["y_pred"].to_numpy().astype(int)

            rows.append(
                {
                    "base": base_name,
                    "fault_line": _safe_int(line),
                    "n_samples": int(n),
                    "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
                }
            )

        # worst first
        rows.sort(key=lambda r: r["macro_f1"])
        return rows

    if fault_line_col in oof.columns:
        bases: Dict[str, pd.DataFrame] = {}

        # all windows base (optional, but recommended to keep explicit)
        if log_all_windows_fault_line:
            bases["all_windows"] = oof

        # status-based base
        if log_fault_start_only and (fault_start_col in oof.columns):
            bases["fault_start_only"] = oof[oof[fault_start_col].astype(str) == fault_start_value]

        # label-based base (exclude no_fault)
        if log_fault_only:
            bases["fault_only"] = oof[
                oof["y_true"].to_numpy().astype(int) != int(no_fault_class_id)
            ]

        # combined base (often the most meaningful for line effects)
        if log_fault_only and log_fault_start_only and (fault_start_col in oof.columns):
            bases["fault_only_fault_start"] = oof[
                (oof["y_true"].to_numpy().astype(int) != int(no_fault_class_id))
                & (oof[fault_start_col].astype(str).to_numpy() == fault_start_value)
            ]

        per_base_rows: Dict[str, List[Dict[str, Any]]] = {}

        for base_name, base_df in bases.items():
            rows = _per_fault_line_f1(base_df, base_name=base_name)
            per_base_rows[base_name] = rows
            try:
                wandb.log({f"{prefix}/{base_name}/f1_by_fault_line": _as_wandb_table(rows)})
            except Exception as e:
                logger.warning(f"Failed to log f1_by_fault_line for base {base_name}: {e}")

        results["f1_by_fault_line"] = per_base_rows

    return results


# =============================================================================
# 5) Regression post-hoc (per-fault-line and per-fault-class analyses)
# =============================================================================


def regression_posthoc_analysis(
    oof: pd.DataFrame,
    *,
    prefix: str = "cv/oof",
    fault_line_col: str = "y_fault_line",
    fault_start_col: str = "status",
    fault_start_value: str = "fault_start",
    min_support_group: int = 30,
    enforce_fault_only: bool = True,
    log_tables: bool = True,
    log_flat_summary: bool = True,
) -> Dict[str, Any]:
    """
    Regression posthoc analysis for fault-only regression targets (e.g., fault location).
    Assumption: regression is defined only on fault windows. If 'status' exists, we filter to fault_start.
    """

    _require_cols(oof, ["y_true", "y_pred"], "regression_posthoc_analysis")

    # ---------------------------------------------------------------------
    # 1) Restrict to fault windows (if status column exists)
    # ---------------------------------------------------------------------
    df = oof
    if fault_start_col in df.columns:
        df = df[df[fault_start_col].astype(str) == fault_start_value].copy()

    if enforce_fault_only:
        if len(df) == 0:
            raise ValueError(
                f"regression_posthoc_analysis: no rows left after fault-only filtering "
                f"({fault_start_col} == '{fault_start_value}')."
            )
        if "fault_class" in df.columns:
            # common convention: 0 == no_fault
            # If you ever include no_fault windows in regression, that is a dataset bug.
            n_no_fault = int((df["fault_class"].astype(int) == 0).sum())
            if n_no_fault > 0:
                raise ValueError(
                    f"regression_posthoc_analysis: found {n_no_fault} rows with fault_class==0 "
                    "(no_fault) in regression OOF data. Regression should be fault-only."
                )

    # ---------------------------------------------------------------------
    # 2) Helpers: safe numeric extraction
    # ---------------------------------------------------------------------
    def _finite_yt_yp(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        mask = np.isfinite(yt) & np.isfinite(yp)
        return yt[mask], yp[mask]

    def _summary(g: pd.DataFrame) -> Dict[str, float]:
        yt, yp = _finite_yt_yp(g)
        if yt.size == 0:
            return {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "median_abs_error": float("nan"),
                "p90_abs_error": float("nan"),
                "n_samples": 0,
            }

        ae = np.abs(yt - yp)
        return {
            "n_samples": int(yt.size),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(root_mean_squared_error(yt, yp)),
            "r2": float(r2_score(yt, yp)),
            "median_abs_error": float(np.median(ae)),
            "p90_abs_error": float(np.percentile(ae, 90)),
        }

    def _per_group(
        g: pd.DataFrame,
        *,
        group_col: str,
        group_name: str,
        label_map: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, Any]]:
        if group_col not in g.columns:
            return []

        rows: List[Dict[str, Any]] = []

        # explicit observed=False avoids pandas FutureWarning if categorical
        for key, gg in g.groupby(group_col, dropna=False, observed=False):
            yt, yp = _finite_yt_yp(gg)
            n = int(yt.size)
            if n < min_support_group:
                continue

            ae = np.abs(yt - yp)

            # format group key + optional label
            key_int: Optional[int] = None
            try:
                key_int = int(np.asarray(key).item())
            except Exception:
                key_int = None

            row: Dict[str, Any] = {
                group_name: key_int if key_int is not None else str(key),
                "n_samples": n,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(root_mean_squared_error(yt, yp)),
                "r2": float(r2_score(yt, yp)),
                "median_abs_error": float(np.median(ae)),
                "p90_abs_error": float(np.percentile(ae, 90)),
            }

            if label_map is not None and key_int is not None:
                row[f"{group_name}_label"] = label_map.get(key_int, str(key_int))

            rows.append(row)

        # worst first
        rows.sort(key=lambda r: r["mae"], reverse=True)
        return rows

    # ---------------------------------------------------------------------
    # 3) Compute summaries
    # ---------------------------------------------------------------------
    global_summary = _summary(df)

    results: Dict[str, Any] = {
        "global_summary": global_summary,
        "n_rows_input": int(len(oof)),
        "n_rows_fault_only": int(len(df)),
    }

    # ---------------------------------------------------------------------
    # 4) W&B logging
    # ---------------------------------------------------------------------
    # (A) hierarchical dict (nice to view)
    wandb.log({f"{prefix}/regression_error_summary": global_summary})

    # (B) flat scalars (nice to sort/filter)
    if log_flat_summary:
        flat = {
            f"{prefix}/reg/mae": global_summary["mae"],
            f"{prefix}/reg/rmse": global_summary["rmse"],
            f"{prefix}/reg/r2": global_summary["r2"],
            f"{prefix}/reg/p90_abs_error": global_summary["p90_abs_error"],
            f"{prefix}/reg/n_samples": global_summary["n_samples"],
            f"{prefix}/reg/n_rows_fault_only": int(len(df)),
        }
        wandb.log(flat)
        for k, v in flat.items():
            wandb.summary[k] = v

    # ---------------------------------------------------------------------
    # 5) Per-fault-line + per-fault-class tables
    # ---------------------------------------------------------------------
    rows_line = _per_group(df, group_col=fault_line_col, group_name="fault_line", label_map=None)
    results["error_by_fault_line_rows"] = rows_line
    if log_tables and rows_line:
        wandb.log({f"{prefix}/error_by_fault_line": _as_wandb_table(rows_line)})

    rows_class = _per_group(
        df,
        group_col="fault_class",
        group_name="fault_class_id",
        label_map=FAULT_ID_TO_LABEL,  # adds fault_class_id_label
    )
    results["error_by_fault_class_rows"] = rows_class
    if log_tables and rows_class:
        wandb.log({f"{prefix}/error_by_fault_class": _as_wandb_table(rows_class)})

    return results


# =============================================================================
# Dispatcher
# =============================================================================


def run_posthoc_analysis(
    oof: pd.DataFrame,
    *,
    task_type: str,
    prefix: str = "cv/oof",
    enforce_all_known_classes: bool = False,
) -> None:
    """
    Call once after you construct `oof` from your CV loop.
    For classification: assumes y_true/y_pred are FAULT_LABEL_TO_ID integers already.
    """
    if task_type in ("binary", "multiclass"):
        classification_posthoc_analysis(
            oof,
            prefix=prefix,
            enforce_all_known_classes=enforce_all_known_classes,
        )
    elif task_type == "regression":
        regression_posthoc_analysis(
            oof=oof,
            prefix=prefix,
        )
    else:
        raise ValueError(f"Unknown task_type={task_type}")


def get_ablation_tag(config: MainConfig) -> str:
    if not getattr(config, "ablation", None) or not config.ablation.enabled:
        return "full"
    if config.ablation.mode == "full":
        return "full"
    if config.ablation.mode == "single_relay":
        return f"single_relay_{config.ablation.relay_index}"
    return "unknown_ablation"


# =============================================================================
# Download to for offline analysis
# =============================================================================
def get_run_out_dir(config: MainConfig) -> Path:
    # Deterministic-ish folder name (good enough for manual interactive runs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cwd = os.getcwd()
    ablation_tag = get_ablation_tag(config)
    out_dir = (
        Path(cwd)
        / "outputs"
        / f"{config.dataset.topology}"
        / f"{config.training.target_label}"
        / f"{config.model.model_name}"
        / f"window{config.window_extraction.window_length}s_step{config.window_extraction.step_length_seconds}s"
        / f"ablation_{ablation_tag}"
        / f"run_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Offline outputs will be written to: {out_dir}")
    return out_dir


def save_offline_outputs(
    out_dir: Path,
    oof_df: pd.DataFrame,
    fold_metrics: list[dict],
    config: MainConfig,
) -> None:
    # ---- 1) Parquet for offline analysis (recommended) ----
    oof_path = out_dir / "oof_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)

    # ---- 2) Optional: arrays-only dump ----
    npz_path = out_dir / "oof_predictions.npz"
    np.savez_compressed(
        npz_path,
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        fold=oof_df["fold"].to_numpy(),
    )

    # ---- 3) Save metrics + config snapshot ----
    (out_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
    # Hydra configs are dataclasses/omegaconf; easiest: just dump resolved YAML via OmegaConf if available
    try:
        from omegaconf import OmegaConf

        (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    except Exception:
        # fallback: don't fail the run because of config serialization
        pass

    logger.info(f"Saved offline outputs to: {out_dir}")
    logger.info(f"  - {oof_path}")
    logger.info(f"  - {npz_path}")
