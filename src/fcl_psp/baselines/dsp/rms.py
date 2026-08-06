# Vendored from protect-90-baselines (package protect90_baselines), signal/rms.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Windowed true RMS.

Registry: see docs/formula_registry.md §2.

    X_rms = sqrt( (1/N) * sum_{k=0}^{N-1} x[k]^2 )

Computed over the last full cycle by default so it is comparable with the
fundamental phasor magnitude. Note this is a *true* RMS: it includes DC and
harmonics, unlike the fundamental-only phasor magnitude.
"""

from __future__ import annotations

import numpy as np

# (provenance cite removed during vendoring)
REFERENCE = "rms"


def rms(
    window: np.ndarray,
    samples_per_cycle: int | None = None,
    *,
    cycle: str = "last",
) -> np.ndarray:
    """True RMS of ``window`` over one cycle (or the whole window if N is None).

    ``window`` is ``(T,)`` or ``(T, M)``; returns a scalar or ``(M,)`` array.
    """
    w = np.asarray(window, dtype=float)
    if samples_per_cycle is not None:
        n = int(samples_per_cycle)
        if w.shape[0] < n:
            raise ValueError(f"window has {w.shape[0]} samples < one cycle ({n})")
        seg = w[-n:] if cycle == "last" else w[:n]
    else:
        seg = w
    return np.sqrt(np.mean(seg**2, axis=0))
