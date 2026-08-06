# Vendored from protect-90-baselines (package protect90_baselines), algorithms/fault_classification.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Rule-based fault classification (phase selection + ground involvement).

Registry: docs/formula_registry.md (rule-based fault classification). Label
`simplified_baseline`.

Phase selection: a phase is faulted if its fundamental current rises above
``phase_pickup_ratio`` times its pre-fault (clean-window) baseline:

    faulted_p = |I_p| > phase_pickup_ratio * I_baseline_p

Ground involvement: residual current ``|3 I_0| = |I_a + I_b + I_c|`` exceeds
``ground_residual_ratio`` times the positive-sequence current:

    grounded = |I_a + I_b + I_c| > ground_residual_ratio * |I_1|

The fault-type label is built from the selected phases and ground flag, e.g.
``AG``, ``BC``, ``ABG``, ``ABC``.
"""

from __future__ import annotations

import numpy as np

# (provenance cite removed during vendoring)
REFERENCE = "fault_classification"

_PHASE_LETTERS = ("A", "B", "C")


def select_faulted_phases(
    i_mag: np.ndarray,
    baseline_i: np.ndarray,
    pickup_ratio: float,
    *,
    min_baseline: float = 1e-6,
) -> list[int]:
    """Indices of faulted phases (current risen above pickup * baseline)."""
    i_mag = np.asarray(i_mag, dtype=float)
    base = np.maximum(np.asarray(baseline_i, dtype=float), min_baseline)
    return [p for p in range(3) if i_mag[p] > pickup_ratio * base[p]]


def ground_involved(
    i_residual: complex,
    i1: complex,
    residual_ratio: float,
    *,
    min_i1: float = 1e-6,
) -> bool:
    """Whether the residual current indicates ground involvement."""
    return abs(i_residual) > residual_ratio * max(abs(i1), min_i1)


def fault_type_from_selection(phases: list[int], grounded: bool) -> str:
    """Build the canonical fault-type label from selected phases + ground."""
    if not phases:
        return "none"
    label = "".join(_PHASE_LETTERS[p] for p in sorted(phases))
    # A balanced three-phase fault is reported as ABC (ground flag ignored:
    # 3-phase faults are symmetric and the residual is ~0 by construction).
    if grounded and len(phases) < 3:
        label += "G"
    return label
