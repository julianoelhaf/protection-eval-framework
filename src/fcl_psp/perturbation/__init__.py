"""Measurement-fidelity perturbation operators for Experiment B (IJEPES revision).

Window-level, test-fold-only perturbations applied to the raw waveform windows
*before* standardization: additive Gaussian noise (SNR sweep), a CT-saturation
proxy (current channels only), and synchronization jitter (per-relay time
offset). See ``agent/coding_plan.md`` (Experiment B) for the design.

All operators are pure functions on ``(N, L, F)`` float32 arrays, never mutate
their input, and take an explicit ``numpy.random.Generator`` for reproducibility.
Channel addressing is resolved from ``feature_names`` (authoritative), never from
the ``psp_helper.constants`` current/voltage index constants (which disagree with
the actual dataset's channel order — currents are channels 0-2, voltages 3-5).
"""

from fcl_psp.perturbation.operators import (
    ChannelLayout,
    add_gaussian_snr,
    apply_axis,
    apply_ct_saturation,
    apply_sync_jitter,
    channel_layout,
)
from fcl_psp.perturbation.seeding import AXIS_ID, make_rng

__all__ = [
    "ChannelLayout",
    "channel_layout",
    "add_gaussian_snr",
    "apply_ct_saturation",
    "apply_sync_jitter",
    "apply_axis",
    "make_rng",
    "AXIS_ID",
]
