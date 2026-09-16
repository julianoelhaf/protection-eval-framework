# fcl_psp/models/posthoc_runner.py

from __future__ import annotations

from typing import Literal

import pandas as pd

from fcl_psp.models.posthoc_analysis import run_posthoc_analysis
from fcl_psp.models.selection_metrics import log_classification_selection_metrics

TaskType = Literal["binary", "multiclass", "regression"]


def run_posthoc_from_oof(
    *,
    oof_df: pd.DataFrame,
    task_type: TaskType,
    prefix: str = "cv/oof",
    enforce_all_known_classes: bool = True,
    log_selection_metrics: bool = True,
) -> None:
    """
    Runs:
      - selection metrics (classification only; optional)
      - posthoc analysis (classification or regression)
    """
    if task_type in ("binary", "multiclass") and log_selection_metrics:
        log_classification_selection_metrics(oof_df, prefix=prefix)

    run_posthoc_analysis(
        oof=oof_df,
        task_type=task_type,
        prefix=prefix,
        enforce_all_known_classes=enforce_all_known_classes,
    )
