"""Unit tests for the fault-label construction logic in ``fcl_psp.models.run_model``.

These exercise pure functions only (no dataset, no model training) and validate
the label string format, the no-fault path, the error path, and consistency with
the canonical ``FAULT_LABEL_TO_ID`` map from ``psp_helper``.
"""

import pandas as pd
import pytest
from psp_helper.constants import FAULT_LABEL_TO_ID

from fcl_psp.models.run_model import build_fault_label, create_fault_classes


def test_build_fault_label_single_phase_to_ground():
    label = build_fault_label(
        event_type="flt_1phg_shc",
        a=True,
        b=False,
        c=False,
        is_grounded=True,
        status="fault_start",
    )
    assert label == "flt_1phg_shc_AG"
    assert label in FAULT_LABEL_TO_ID


def test_build_fault_label_three_phase_no_ground():
    label = build_fault_label(
        event_type="flt_3ph_shc",
        a=True,
        b=True,
        c=True,
        is_grounded=False,
        status="fault_start",
    )
    assert label == "flt_3ph_shc_ABC"
    assert label in FAULT_LABEL_TO_ID


def test_non_fault_start_window_is_no_fault():
    # Any window that does not contain the fault inception is labelled no_fault,
    # regardless of the phase flags.
    label = build_fault_label(
        event_type="flt_1phg_shc",
        a=True,
        b=True,
        c=True,
        is_grounded=True,
        status="post_fault",
    )
    assert label == "no_fault"


def test_fault_start_without_phase_raises():
    with pytest.raises(ValueError):
        build_fault_label(
            event_type="flt_1phg_shc",
            a=False,
            b=False,
            c=False,
            is_grounded=False,
            status="fault_start",
        )


def test_create_fault_classes_maps_to_ids():
    labels = pd.DataFrame(
        {
            "event_type": ["flt_1phg_shc", "flt_3ph_shc", "flt_1phg_shc"],
            "y_phase_A": [1, 1, 0],
            "y_phase_B": [0, 1, 0],
            "y_phase_C": [0, 1, 0],
            "y_is_grounded": [1, 0, 0],
            "status": ["fault_start", "fault_start", "post_fault"],
        }
    )
    ids = create_fault_classes(labels)
    assert ids == [
        FAULT_LABEL_TO_ID["flt_1phg_shc_AG"],
        FAULT_LABEL_TO_ID["flt_3ph_shc_ABC"],
        FAULT_LABEL_TO_ID["no_fault"],
    ]


def test_create_fault_classes_missing_columns_raises():
    labels = pd.DataFrame({"event_type": ["flt_1phg_shc"]})
    with pytest.raises(ValueError):
        create_fault_classes(labels)
