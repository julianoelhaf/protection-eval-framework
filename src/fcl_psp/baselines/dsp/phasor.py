# Vendored from protect-90-baselines (package protect90_baselines), signal/phasor.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Fundamental phasor extraction by a full-cycle DFT.

Registry: see docs/formula_registry.md §1.

    X_1 = (2/N) * sum_{k=0}^{N-1} x[k] * exp(-j 2 pi k / N)

The 2/N scaling yields **peak-amplitude** phasors. Use ``scaling="rms"`` to get
RMS phasors (peak / sqrt(2)). The last full cycle of the window is used by
default, assuming the fault is developed there.
"""

from __future__ import annotations

import numpy as np

# (provenance cite removed during vendoring)
REFERENCE = "dft_phasor"

_SQRT2 = np.sqrt(2.0)


def fundamental_phasors(
    window: np.ndarray,
    samples_per_cycle: int,
    *,
    scaling: str = "peak",
    cycle: str = "last",
) -> np.ndarray:
    """Per-signal fundamental phasors from one cycle of ``window``.

    Parameters
    ----------
    window : array ``(T,)`` or ``(T, M)``
        Real samples; ``M`` independent signals (e.g. 3 phases).
    samples_per_cycle : int
        Number of samples in one fundamental cycle (N).
    scaling : {"peak", "rms"}
        Peak (2/N) or RMS (2/N / sqrt(2)) phasor magnitude.
    cycle : {"last", "first"}
        Which full cycle of the window to transform.

    Returns
    -------
    complex scalar (1-D input) or complex array ``(M,)`` (2-D input).
    """
    w = np.asarray(window, dtype=float)
    n = int(samples_per_cycle)
    if n <= 0:
        raise ValueError(f"samples_per_cycle must be positive, got {n}")
    if w.shape[0] < n:
        raise ValueError(f"window has {w.shape[0]} samples < one cycle ({n})")
    if cycle == "last":
        seg = w[-n:]
    elif cycle == "first":
        seg = w[:n]
    else:
        raise ValueError(f"unknown cycle {cycle!r}; expected 'last' or 'first'")
    k = np.exp(-1j * 2 * np.pi * np.arange(n) / n)
    phasor = (2.0 / n) * (seg.T @ k)
    if scaling == "rms":
        phasor = phasor / _SQRT2
    elif scaling != "peak":
        raise ValueError(f"unknown scaling {scaling!r}; expected 'peak' or 'rms'")
    return phasor


def phasor_magnitude(window: np.ndarray, samples_per_cycle: int, **kw: str) -> np.ndarray:
    """Magnitude of the fundamental phasor(s)."""
    return np.abs(fundamental_phasors(window, samples_per_cycle, **kw))
