"""Deterministic per-(fold, axis, level, realization) RNG derivation.

A single ``SEED_GLOBAL`` deterministically seeds every stochastic draw in
Experiment B via ``numpy.random.SeedSequence``, giving collision-free,
independent streams. This answers reviewer R1.11 (seed management) concretely
and makes the perturbation curves exactly reproducible.

The repository sets no global numpy/torch seed (only the sklearn estimator
``random_state=42``), so this scheme is self-contained and does not interfere
with model determinism.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

# Stable integer id per perturbation axis. Extend (do not renumber) if axes are
# added, so previously generated realizations stay reproducible.
AXIS_ID = {
    "noise": 1,
    "ct_saturation": 2,
    "jitter": 3,
    "train_aug": 4,
}

Level = Union[int, float, str, None]


def _level_id(level: Level) -> int:
    """Map a perturbation level to a stable non-negative integer.

    Levels come from YAML as ints (jitter samples), floats (CT fraction), or the
    string sentinels ``"clean"`` / ``"no_sat"`` (the identity level). The exact
    mapping does not matter as long as it is deterministic and distinct across
    the levels used in one axis; we hash the canonical string form.
    """
    if level is None:
        return 0
    if isinstance(level, str):
        if level.lower() in ("clean", "no_sat", "none"):
            return 0
        s = level
    elif isinstance(level, bool):  # guard: bool is an int subclass
        s = str(int(level))
    elif isinstance(level, (int, float)):
        # Encode with fixed precision so 0.7 and 0.70 collapse to one id.
        s = f"{float(level):.6g}"
    else:
        s = str(level)
    # Deterministic, process-independent small hash (avoid Python's salted hash()).
    acc = 0
    for ch in s:
        acc = (acc * 131 + ord(ch)) % 2_000_003
    return acc + 1  # +1 so a non-clean level never collides with the clean id 0


def make_rng(
    seed_global: int,
    fold: int,
    axis: str,
    level: Level,
    rep: int,
) -> np.random.Generator:
    """Return an independent ``Generator`` for one (fold, axis, level, rep) cell.

    ``SeedSequence`` spawns statistically independent streams from the integer
    entropy tuple, so different cells never share draws and the same cell always
    reproduces the same draws.
    """
    if axis not in AXIS_ID:
        raise ValueError(f"Unknown perturbation axis {axis!r}; known: {sorted(AXIS_ID)}")
    ss = np.random.SeedSequence(
        [int(seed_global), AXIS_ID[axis], _level_id(level), int(fold), int(rep)]
    )
    return np.random.default_rng(ss)


def seed_entropy(seed_global: int, fold: int, axis: str, level: Level, rep: int) -> Optional[list]:
    """The entropy tuple used for a cell (for logging/provenance)."""
    return [int(seed_global), AXIS_ID[axis], _level_id(level), int(fold), int(rep)]
