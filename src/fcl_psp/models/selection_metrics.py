from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import wandb
from psp_helper.constants import FAULT_ID_TO_LABEL
from sklearn.metrics import f1_score, precision_score, recall_score


def log_classification_selection_metrics(
    oof: pd.DataFrame,
    *,
    prefix: str = "cv/oof",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    status_col: str = "status",
    fault_start_value: str = "fault_start",
    no_fault_id: int = 0,
) -> Dict[str, float]:
    """
    Logs (and returns) three robust classification metrics:
      1) macro-F1 on ALL windows (includes no_fault)
      2) macro-F1 on fault_start only (status == fault_start_value)
      3) macro-F1 excluding no_fault label (computed on all windows, but labels exclude 0)

    Also logs macro precision/recall for the same three views (useful context).
    """
    # ---- Basic checks ----
    for c in (y_true_col, y_pred_col):
        if c not in oof.columns:
            raise ValueError(
                f"Missing required column '{c}' in oof. Found: {oof.columns.tolist()}"
            )

    y_true = oof[y_true_col].to_numpy().astype(int)
    y_pred = oof[y_pred_col].to_numpy().astype(int)

    # Determine fault IDs (exclude no_fault)
    all_ids = sorted(int(k) for k in FAULT_ID_TO_LABEL.keys())
    fault_ids = np.array([k for k in all_ids if k != no_fault_id], dtype=int)

    out: Dict[str, float] = {}

    def _macro_triplet(yt: np.ndarray, yp: np.ndarray, *, labels: Optional[np.ndarray] = None):
        return (
            float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            float(precision_score(yt, yp, labels=labels, average="macro", zero_division=0)),
            float(recall_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        )

    # 1) Global (includes no_fault)
    f1_all, p_all, r_all = _macro_triplet(y_true, y_pred, labels=np.array(all_ids, dtype=int))
    out[f"{prefix}/sel/f1_macro_all"] = f1_all
    out[f"{prefix}/sel/precision_macro_all"] = p_all
    out[f"{prefix}/sel/recall_macro_all"] = r_all

    # 2) Fault-start only (if status exists)
    if status_col in oof.columns:
        mask_fs = oof[status_col].astype(str).to_numpy() == fault_start_value
        if mask_fs.any():
            f1_fs, p_fs, r_fs = _macro_triplet(
                y_true[mask_fs],
                y_pred[mask_fs],
                labels=np.array(all_ids, dtype=int),  # keep full axis stable if you want
            )
            out[f"{prefix}/sel/f1_macro_fault_start_only"] = f1_fs
            out[f"{prefix}/sel/precision_macro_fault_start_only"] = p_fs
            out[f"{prefix}/sel/recall_macro_fault_start_only"] = r_fs
        else:
            # still log something predictable
            out[f"{prefix}/sel/f1_macro_fault_start_only"] = float("nan")
            out[f"{prefix}/sel/precision_macro_fault_start_only"] = float("nan")
            out[f"{prefix}/sel/recall_macro_fault_start_only"] = float("nan")
    else:
        out[f"{prefix}/sel/f1_macro_fault_start_only"] = float("nan")
        out[f"{prefix}/sel/precision_macro_fault_start_only"] = float("nan")
        out[f"{prefix}/sel/recall_macro_fault_start_only"] = float("nan")

    # 3) Exclude no_fault (labels exclude 0, computed over all windows)
    f1_nf, p_nf, r_nf = _macro_triplet(y_true, y_pred, labels=fault_ids)
    out[f"{prefix}/sel/f1_macro_excl_no_fault"] = f1_nf
    out[f"{prefix}/sel/precision_macro_excl_no_fault"] = p_nf
    out[f"{prefix}/sel/recall_macro_excl_no_fault"] = r_nf

    # ---- Log to W&B ----
    # Flatten keys a bit for W&B; you can also log as a single dict.
    wandb.log(out)

    # Also push to summary so it's easy to sort runs by the metric you care about
    for k, v in out.items():
        wandb.summary[k] = v

    return out
