# Vendored from protect-90-baselines (package protect90_baselines), signal/relay_features.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Per-relay phasor/sequence features for one window.

A *relay* here is one line terminal (8 in total). For each relay we compute the
fundamental phasor of the 3 phase currents and 3 phase voltages, then their
sequence components. These features feed the detection / classification /
faulted-line-identification baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phasor import fundamental_phasors
from .sequence import sequence_components


@dataclass
class RelayFeatures:
    name: str
    line: str
    terminal: str
    i_abc: np.ndarray  # complex (3,)
    v_abc: np.ndarray  # complex (3,)
    i0: complex
    i1: complex
    i2: complex
    v0: complex
    v1: complex
    v2: complex

    @property
    def i_mag(self) -> np.ndarray:
        return np.abs(self.i_abc)

    @property
    def v_mag(self) -> np.ndarray:
        return np.abs(self.v_abc)

    @property
    def i_residual(self) -> complex:
        return complex(np.sum(self.i_abc))  # = 3*I0


def relay_features_from_phasors(
    phasors: np.ndarray,
    relays: list[dict],
) -> list[RelayFeatures]:
    """Build :class:`RelayFeatures` from a length-48 phasor vector.

    Use with :func:`fcl_psp.baselines.signal.phasor.fundamental_phasors` applied
    once to the whole window (all 48 channels) — much faster than per-relay DFTs.
    """
    feats = []
    for r in relays:
        i_abc = np.asarray(phasors)[r["cur"]]
        v_abc = np.asarray(phasors)[r["vol"]]
        i0, i1, i2 = sequence_components(i_abc)
        v0, v1, v2 = sequence_components(v_abc)
        feats.append(
            RelayFeatures(
                name=r["name"],
                line=r["line"],
                terminal=r["terminal"],
                i_abc=i_abc,
                v_abc=v_abc,
                i0=i0,
                i1=i1,
                i2=i2,
                v0=v0,
                v1=v1,
                v2=v2,
            )
        )
    return feats


def relay_features(
    window: np.ndarray,
    relays: list[dict],
    samples_per_cycle: int,
    *,
    scaling: str = "peak",
    cycle: str = "last",
) -> list[RelayFeatures]:
    """Compute :class:`RelayFeatures` for every relay (one DFT over all channels)."""
    phasors = fundamental_phasors(window, samples_per_cycle, scaling=scaling, cycle=cycle)
    return relay_features_from_phasors(phasors, relays)
