"""Integration tests for the conventional-baseline estimators + vendored algorithms.

Dataset-free: synthetic phasors with known ground truth. Validates the label
mapping (conventional FC classes == ML FC classes) and exact distance recovery
of the two impedance locators under ideal (no-infeed) conditions.
"""

import numpy as np
import pytest
from psp_helper.constants import FAULT_LABEL_TO_ID

from fcl_psp.baselines.algorithms.fault_location import two_ended_positive_sequence
from fcl_psp.baselines.algorithms.single_ended import single_ended_reactance
from fcl_psp.baselines.estimators import _code_to_id_table, phases_ground_to_id


# --- FC label mapping parity with the ML task ---
@pytest.mark.parametrize(
    "phases,grounded,label",
    [
        ([0], True, "flt_1phg_shc_AG"),
        ([1], True, "flt_1phg_shc_BG"),
        ([0, 1], False, "flt_2ph_shc_AB"),
        ([1, 2], False, "flt_2ph_shc_BC"),
        ([0, 1], True, "flt_2phg_shc_ABG"),
        ([0, 1, 2], False, "flt_3ph_shc_ABC"),
    ],
)
def test_phases_ground_to_id_matches_framework_labels(phases, grounded, label):
    assert phases_ground_to_id(phases, grounded) == FAULT_LABEL_TO_ID[label]


def test_empty_phase_set_is_no_fault():
    assert phases_ground_to_id([], False) == FAULT_LABEL_TO_ID["no_fault"]
    assert phases_ground_to_id([], True) == FAULT_LABEL_TO_ID["no_fault"]


def test_code_to_id_table_shape_and_ids_valid():
    table = _code_to_id_table()
    assert table.shape == (16,)
    valid = set(FAULT_LABEL_TO_ID.values())
    assert set(table.tolist()).issubset(valid)
    assert table[0] == FAULT_LABEL_TO_ID["no_fault"]  # no phase picked -> no_fault


# --- two-ended locator: exact recovery of a known distance ---
@pytest.mark.parametrize("m", [0.05, 0.25, 0.5, 0.73, 0.95])
def test_two_ended_recovers_distance(m):
    z1 = complex(3.0, 12.0)  # total line series impedance
    i_s = complex(1.0, -0.3)  # arbitrary terminal currents (into the line)
    i_r = complex(0.8, 0.2)
    v_fault = complex(50.0, 5.0)  # voltage at the fault point
    v_s = v_fault + m * z1 * i_s  # V_S = V_fault + m Z1 I_S
    v_r = v_fault + (1 - m) * z1 * i_r
    est = two_ended_positive_sequence(v_s, i_s, v_r, i_r, z1)
    assert est == pytest.approx(m, abs=1e-9)


# --- single-ended reactance: exact recovery on an ideal phase-phase loop ---
@pytest.mark.parametrize("m", [0.1, 0.4, 0.8])
def test_single_ended_phase_phase_recovers_distance(m):
    z1 = complex(0.0, 10.0)  # purely reactive line so reactance fraction is exact
    # phase-phase (BC) loop: V_p - V_q = m Z1 (I_p - I_q)
    i = [complex(0.0, 0.0), complex(1.0, -0.2), complex(-0.7, 0.4)]  # A,B,C currents
    dv = m * z1 * (i[1] - i[2])
    v = [complex(0, 0), complex(dv, 0), complex(0, 0)]  # only the B-C difference matters
    dist, mode = single_ended_reactance(v, i, z1, faulted_phases=[1, 2], grounded=False)
    assert mode == "phase_phase"
    assert dist == pytest.approx(m, abs=1e-9)


def test_two_ended_degenerate_returns_nan():
    z1 = complex(3.0, 12.0)
    # I_S + I_R ~ 0 -> degenerate denominator
    out = two_ended_positive_sequence(1 + 1j, complex(1, 0), 1 + 1j, complex(-1, 0), z1)
    assert np.isnan(out)
