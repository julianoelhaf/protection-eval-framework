"""Typed config for Experiment B perturbation sweeps.

Loaded *outside* ``psp_helper.config.MainConfig`` (which is an external
structured schema that would reject unknown groups). The evaluator builds this
via ``OmegaConf.structured(PerturbationConfig)`` merged with
``config/perturbation/default.yaml`` and any CLI overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Union


@dataclass
class NoiseAxis:
    # dB SNR levels; the string "clean" is the identity level (leftmost anchor).
    # List[Any]: OmegaConf structured configs reject mixed str/int/float under a
    # Union element type, so the element type is left open.
    snr_db: List[Any] = field(default_factory=lambda: ["clean", 40, 30, 20, 10])


@dataclass
class CtSaturationAxis:
    # Retained fraction of clean peak on current channels; "no_sat" = identity.
    c: List[Any] = field(default_factory=lambda: ["no_sat", 0.7, 0.5, 0.3])


@dataclass
class JitterAxis:
    # Per-relay max offset in samples; 0 = identity. {0,1,2,4} = {0,156,312,625} us @ 6400 Hz.
    delta_max_samples: List[int] = field(default_factory=lambda: [0, 1, 2, 4])


@dataclass
class TrainNoisy:
    # B.5 mitigation: retrain with noise mixed into training, then evaluate on test.
    enabled: bool = False
    axis: str = "noise"
    level: Any = 20
    mix: float = 0.5  # fraction of clean (un-perturbed) training samples retained


@dataclass
class PerturbationConfig:
    enabled: bool = True
    seed_global: int = 12345
    n_realizations: int = 5  # R
    horizons_ms: List[int] = field(default_factory=lambda: [20, 50])
    axes: List[str] = field(default_factory=lambda: ["noise", "ct_saturation", "jitter"])
    noise: NoiseAxis = field(default_factory=NoiseAxis)
    ct_saturation: CtSaturationAxis = field(default_factory=CtSaturationAxis)
    jitter: JitterAxis = field(default_factory=JitterAxis)
    train_noisy: TrainNoisy = field(default_factory=TrainNoisy)
    # Output directory for perturbation metrics (created if missing).
    out_dir: str = "reports/perturbation"

    def levels_for(self, axis: str) -> List[Union[str, float, int]]:
        if axis in ("noise", "train_aug"):
            return list(self.noise.snr_db)
        if axis == "ct_saturation":
            return list(self.ct_saturation.c)
        if axis == "jitter":
            return list(self.jitter.delta_max_samples)
        raise ValueError(f"Unknown axis {axis!r}")
