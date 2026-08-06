"""Unit tests for ``fcl_psp.models.selection_metrics``.

The function under test mixes metric computation with W&B logging, so we stub out
``wandb.log``/``wandb.summary`` and assert on the returned dict. Covers the perfect
prediction case, the fault-start-only subset, and the NaN-fallback branches.
"""

import math

import pandas as pd
import pytest
from psp_helper.constants import FAULT_ID_TO_LABEL

import fcl_psp.models.selection_metrics as sm


@pytest.fixture(autouse=True)
def _stub_wandb(monkeypatch):
    monkeypatch.setattr(sm.wandb, "log", lambda *a, **k: None)
    monkeypatch.setattr(sm.wandb, "summary", {})


def _all_ids():
    return sorted(int(k) for k in FAULT_ID_TO_LABEL)


def test_perfect_prediction_gives_unit_macro_f1():
    ids = _all_ids()
    oof = pd.DataFrame(
        {
            "y_true": ids,
            "y_pred": ids,
            "status": ["fault_start"] * len(ids),
        }
    )
    out = sm.log_classification_selection_metrics(oof)
    assert out["cv/oof/sel/f1_macro_all"] == pytest.approx(1.0)
    assert out["cv/oof/sel/f1_macro_excl_no_fault"] == pytest.approx(1.0)
    assert out["cv/oof/sel/f1_macro_fault_start_only"] == pytest.approx(1.0)


def test_imperfect_prediction_below_one():
    ids = _all_ids()
    pred = ids.copy()
    pred[1] = ids[0]  # introduce a single misclassification
    oof = pd.DataFrame({"y_true": ids, "y_pred": pred, "status": ["fault_start"] * len(ids)})
    out = sm.log_classification_selection_metrics(oof)
    assert out["cv/oof/sel/f1_macro_all"] < 1.0


def test_missing_status_column_yields_nan_fault_start():
    ids = _all_ids()
    oof = pd.DataFrame({"y_true": ids, "y_pred": ids})
    out = sm.log_classification_selection_metrics(oof)
    assert math.isnan(out["cv/oof/sel/f1_macro_fault_start_only"])


def test_no_fault_start_rows_yields_nan_fault_start():
    ids = _all_ids()
    oof = pd.DataFrame({"y_true": ids, "y_pred": ids, "status": ["post_fault"] * len(ids)})
    out = sm.log_classification_selection_metrics(oof)
    assert math.isnan(out["cv/oof/sel/f1_macro_fault_start_only"])


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        sm.log_classification_selection_metrics(pd.DataFrame({"y_true": [0, 1]}))
