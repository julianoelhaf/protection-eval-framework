# Vendored from protect-90-baselines (package protect90_baselines), signal/sequence.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Symmetrical-component transform.

Registry: see docs/formula_registry.md §3.

    X_0 = (X_a + X_b + X_c) / 3
    X_1 = (X_a + a X_b + a^2 X_c) / 3
    X_2 = (X_a + a^2 X_b + a X_c) / 3,   a = exp(j 2 pi / 3)

Inputs are per-phase phasors (peak or RMS, consistently scaled), phase order
a-b-c = L1-L2-L3.
"""

from __future__ import annotations

import numpy as np

# (provenance cite removed during vendoring)
REFERENCE = "symmetrical_components"

# Fortescue rotation operator: a = exp(j 2π/3). Exported for use in tests.
A = np.exp(1j * 2 * np.pi / 3)
_A1 = np.array([1.0, A, A**2]) / 3.0  # positive
_A2 = np.array([1.0, A**2, A]) / 3.0  # negative
_A0 = np.array([1.0, 1.0, 1.0]) / 3.0  # zero


def _abc(phasors_abc: np.ndarray) -> np.ndarray:
    p = np.asarray(phasors_abc, dtype=complex)
    if p.shape[-1] != 3:
        raise ValueError(f"expected last axis of length 3 (a,b,c), got {p.shape}")
    return p


def positive_sequence(phasors_abc: np.ndarray) -> complex:
    return complex(_abc(phasors_abc) @ _A1)


def negative_sequence(phasors_abc: np.ndarray) -> complex:
    return complex(_abc(phasors_abc) @ _A2)


def zero_sequence(phasors_abc: np.ndarray) -> complex:
    return complex(_abc(phasors_abc) @ _A0)


def sequence_components(phasors_abc: np.ndarray) -> tuple[complex, complex, complex]:
    """Return ``(X_0, X_1, X_2)`` — zero, positive, negative — for a 3-phase phasor triple.

    Unpack as ``x0, x1, x2 = sequence_components(abc)``.  The ordering follows
    the Fortescue matrix row convention (zero first).
    """
    p = _abc(phasors_abc)
    return complex(p @ _A0), complex(p @ _A1), complex(p @ _A2)


def residual(phasors_abc: np.ndarray) -> complex:
    """Residual current/voltage ``X_a + X_b + X_c = 3 X_0``."""
    return complex(np.sum(_abc(phasors_abc)))
