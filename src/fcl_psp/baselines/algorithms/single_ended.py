# Vendored from protect-90-baselines (package protect90_baselines), algorithms/single_ended.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Single-ended (single-terminal) reactance fault location.

Registry: docs/formula_registry.md (single-ended reactance method). Label
`simplified_baseline`.

Uses only ONE terminal's phase voltages/currents. Fault-loop selection follows
the textbook convention:

* **phase-phase loop** (LL / LLG / 3-phase) -- ``Z_app = (V_p - V_q)/(I_p - I_q)``.
* **phase-ground loop** (single-line-ground, SLG) --
  ``Z_app = V_p / (I_p + k0*I_res + k0m*I_res_parallel)`` with
  ``k0  = (Z0 - Z1)/(3 Z1)``           (zero-sequence self compensation) and
  ``k0m = Z0m/(3 Z1)``                  (zero-sequence MUTUAL compensation).

On PROTECT-90 the two parallel circuits run far apart, so the zero-sequence
**mutual** coupling between them is negligible (``Z0m ≈ 0``; the parameter is also
simply absent from the release). The SLG loop is therefore evaluated with **self
compensation only** by default (``neglect_mutual=True``, mode ``"ground"``) -- this
is the physically correct model here, not a lower bound. If a ``z0m_line`` is
supplied the mutual term is included (mode ``"ground_compensated"``). The only
gating case is a missing ``Z0`` (mode ``"gated"`` -> ``nan``), since the self factor
``k0`` cannot then be formed.

The reactance method takes ``m = Im(Z_app) / Im(Z1)``, cancelling a purely
resistive fault term; it remains sensitive to remote infeed and load (unlike the
two-ended method) -- that, not mutual coupling, is the dominant single-ended error
source on this corridor.

Distance is a fraction from the terminal whose phasors are passed in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# (provenance cite removed during vendoring)
REFERENCE = "single_ended_reactance"


def _reactance_fraction(z_app: complex, z1_line: complex) -> float:
    if abs(z1_line.imag) < 1e-30:
        return float("nan")
    return float(z_app.imag / z1_line.imag)


def single_ended_reactance(
    v_abc: Sequence[complex],
    i_abc: Sequence[complex],
    z1_line: complex,
    faulted_phases: Sequence[int],
    grounded: bool,
    *,
    z0_line: complex | None = None,
    z0m_line: complex | None = None,
    i_res_parallel: complex = 0j,
    neglect_mutual: bool = True,
    min_denominator: float = 1e-30,
) -> tuple[float, str]:
    """Return ``(distance_fraction_from_this_terminal, mode)``.

    ``mode`` is one of ``phase_phase`` | ``ground`` (self-compensated, mutual
    neglected) | ``ground_compensated`` (mutual term included) | ``gated`` (no
    ``Z0`` available, or ``neglect_mutual=False`` with no ``Z0m``). A ``gated`` or
    degenerate result returns ``nan`` for the distance.
    """
    v = np.asarray(v_abc, dtype=complex)
    i = np.asarray(i_abc, dtype=complex)
    fp = [int(p) for p in faulted_phases]

    if len(fp) == 1 and grounded:
        # single-line-ground -> phase-ground loop (zero-sequence handling)
        if z0_line is None:
            return float("nan"), "gated"
        p = fp[0]
        i_res = complex(i.sum())
        k0 = (z0_line - z1_line) / (3.0 * z1_line)
        if z0m_line is not None:
            k0m = z0m_line / (3.0 * z1_line)
            denom = i[p] + k0 * i_res + k0m * complex(i_res_parallel)
            mode = "ground_compensated"
        elif neglect_mutual:
            # Z0m ~ 0 (parallel circuits far apart): self compensation only.
            denom = i[p] + k0 * i_res
            mode = "ground"
        else:
            return float("nan"), "gated"
    else:
        # LL / LLG / 3-phase -> phase-phase loop (zero-sequence free)
        p, q = (fp[0], fp[1]) if len(fp) >= 2 else (0, 1)
        denom = i[p] - i[q]
        mode = "phase_phase"
        if abs(denom) < min_denominator:
            return float("nan"), mode
        return _reactance_fraction((v[p] - v[q]) / denom, z1_line), mode

    if abs(denom) < min_denominator:
        return float("nan"), mode
    return _reactance_fraction(v[p] / denom, z1_line), mode
