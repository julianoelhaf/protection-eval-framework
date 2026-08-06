# Vendored from protect-90-baselines (package protect90_baselines), algorithms/fault_location.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Fault-location algorithms.

Registry: see docs/formula_registry.md §4.

Currently implemented:
* ``two_ended_positive_sequence`` -- synchronized two-terminal positive-sequence
  locator (``textbook_based`` / ``standard_based``). Fault-resistance and infeed
  independent; the strong baseline.

All distances are fractions of line length from the sending terminal S
(= ``line_from``); multiply by 100 for "% of line length".
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..data.channel_map import LineTerminals
from ..data.line_registry import LineParams
from ..dsp.phasor import fundamental_phasors
from ..dsp.sequence import positive_sequence

# (provenance cite removed during vendoring)
REFERENCE = "two_ended_fl"


def two_ended_positive_sequence(
    v_s: complex,
    i_s: complex,
    v_r: complex,
    i_r: complex,
    z1_line: complex,
    *,
    min_current_ratio: float = 1e-2,
) -> float:
    """Synchronized two-ended positive-sequence fault distance (fraction from S).

    Solves ``V_S - m Z1 I_S = V_R - (1-m) Z1 I_R`` (both currents directed INTO
    the line) for ``m = (V_S - V_R + Z1 I_R) / (Z1 (I_S + I_R))``. Returns ``nan``
    if ``|I_S + I_R|`` is negligible relative to the terminal currents (a
    degenerate denominator that would blow up the estimate).
    """
    denom_current = i_s + i_r
    scale = max(abs(i_s), abs(i_r), 1e-30)
    if abs(denom_current) < min_current_ratio * scale:
        return float("nan")
    if abs(z1_line) < 1e-30:
        raise ValueError("z1_line is ~0; cannot locate")
    m = (v_s - v_r + z1_line * i_r) / (z1_line * denom_current)
    return float(m.real)


class TerminalPhasors(NamedTuple):
    """Per-terminal positive-sequence phasors for one window."""

    v_s: complex
    i_s: complex
    v_r: complex
    i_r: complex


def terminal_phasors(
    window: np.ndarray,
    terminals: LineTerminals,
    samples_per_cycle: int,
    *,
    scaling: str = "peak",
    cycle: str = "last",
) -> TerminalPhasors:
    """Extract positive-sequence terminal phasors for a line from one window."""

    def ps(idx: list[int]) -> complex:
        abc = fundamental_phasors(window[:, idx], samples_per_cycle, scaling=scaling, cycle=cycle)
        return positive_sequence(abc)

    return TerminalPhasors(
        v_s=ps(terminals.s_vol),
        i_s=ps(terminals.s_cur),
        v_r=ps(terminals.r_vol),
        i_r=ps(terminals.r_cur),
    )


def locate_two_ended(
    window: np.ndarray,
    terminals: LineTerminals,
    params: LineParams,
    samples_per_cycle: int,
    *,
    min_current_ratio: float = 1e-2,
    current_sign: int = 1,
    scaling: str = "peak",
    cycle: str = "last",
) -> float:
    """Two-ended FL fraction-from-S for one window of a given line.

    ``current_sign`` flips both terminal currents if the dataset's current
    reference is out of the line at one convention; it is determined empirically
    per line in ``validate-channels`` (default +1 = currents into the line).
    """
    tp = terminal_phasors(window, terminals, samples_per_cycle, scaling=scaling, cycle=cycle)
    return two_ended_positive_sequence(
        tp.v_s,
        current_sign * tp.i_s,
        tp.v_r,
        current_sign * tp.i_r,
        params.z1_total,
        min_current_ratio=min_current_ratio,
    )
