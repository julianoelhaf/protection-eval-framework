"""Unit tests for ``fcl_psp.perturbation.operators`` (Experiment B).

Dataset-free: synthetic ``(N, L, F)`` arrays and hand-built ``feature_names``
matching the real PROTECT-90 channel order (currents 0-2, voltages 3-5 per relay).
Mirrors the style of ``tests/test_relay_features.py``.
"""

import numpy as np

from fcl_psp.perturbation.operators import (
    _shift_time_edge,
    add_gaussian_snr,
    apply_ct_saturation,
    apply_sync_jitter,
    channel_layout,
)
from fcl_psp.perturbation.seeding import make_rng

N_RELAYS = 3
L = 128


def _feature_names(n_relays=N_RELAYS):
    # Real order: per relay, currents L1/L2/L3 then voltages L1/L2/L3.
    names = []
    for r in range(n_relays):
        pref = f"Bus_{r}_Line_0{r}_0{r+1}A"
        for ph in ("L1", "L2", "L3"):
            names.append(f"{pref}_cur_{ph}_A")
        for ph in ("L1", "L2", "L3"):
            names.append(f"{pref}_vol_{ph}_V")
    return names


def _windows(n=8, n_relays=N_RELAYS, seed=0):
    rng = np.random.default_rng(seed)
    F = n_relays * 6
    # 50 Hz sinusoids + small offset so RMS is well-defined per channel.
    t = np.arange(L) / 6400.0
    base = np.sin(2 * np.pi * 50 * t)[None, :, None]  # (1,L,1)
    amp = rng.uniform(0.5, 2.0, size=(n, 1, F))
    return (amp * base).astype(np.float32) + rng.standard_normal((n, L, F)).astype(
        np.float32
    ) * 0.01


def test_channel_layout_currents_first():
    lay = channel_layout(_feature_names())
    # currents at 0-2, voltages at 3-5, tiled every 6
    assert list(lay.current_cols) == [0, 1, 2, 6, 7, 8, 12, 13, 14]
    assert list(lay.voltage_cols) == [3, 4, 5, 9, 10, 11, 15, 16, 17]
    assert lay.n_relays == N_RELAYS
    # each relay owns 6 contiguous columns
    assert list(lay.relay_of_col) == [0] * 6 + [1] * 6 + [2] * 6


def test_gaussian_snr_clean_is_identity():
    x = _windows()
    for clean in (None, "clean", "CLEAN"):
        out = add_gaussian_snr(x, clean, make_rng(1, 0, "noise", "clean", 0))
        assert np.array_equal(out, x)
        assert out is not x  # a copy, not the same object


def test_gaussian_snr_hits_target_snr():
    x = _windows(n=64, seed=3)
    target = 20.0
    out = add_gaussian_snr(x, target, make_rng(7, 1, "noise", 20, 0))
    noise = out.astype(np.float64) - x.astype(np.float64)
    p_sig = np.mean(x.astype(np.float64) ** 2, axis=1)  # (N,F)
    p_noise = np.mean(noise**2, axis=1)  # (N,F)
    snr_db = 10 * np.log10(p_sig / p_noise)
    # per (window,channel) achieved SNR should cluster near the target
    assert abs(np.mean(snr_db) - target) < 1.5


def test_gaussian_snr_deterministic():
    x = _windows(seed=5)
    a = add_gaussian_snr(x, 30, make_rng(42, 2, "noise", 30, 1))
    b = add_gaussian_snr(x, 30, make_rng(42, 2, "noise", 30, 1))
    assert np.array_equal(a, b)
    c = add_gaussian_snr(x, 30, make_rng(42, 2, "noise", 30, 2))  # different rep
    assert not np.array_equal(a, c)


def test_ct_saturation_currents_only_and_bounded():
    x = _windows(n=16, seed=9)
    lay = channel_layout(_feature_names())
    c = 0.5
    out = apply_ct_saturation(
        x, c, lay, make_rng(1, 0, "ct_saturation", 0.5, 0), randomize_onset=False
    )
    # voltages untouched
    assert np.array_equal(out[:, :, lay.voltage_cols], x[:, :, lay.voltage_cols])
    # each current channel bounded by c * clean-peak (+ float slack)
    seg_in = x[:, :, lay.current_cols].astype(np.float64)
    peak = np.max(np.abs(seg_in), axis=1, keepdims=True)
    seg_out = out[:, :, lay.current_cols].astype(np.float64)
    assert np.all(np.abs(seg_out) <= c * peak + 1e-4)
    # something actually changed on currents (peaks were clipped)
    assert not np.array_equal(out[:, :, lay.current_cols], x[:, :, lay.current_cols])


def test_ct_saturation_clean_identity():
    x = _windows()
    lay = channel_layout(_feature_names())
    out = apply_ct_saturation(x, "no_sat", lay, make_rng(1, 0, "ct_saturation", "no_sat", 0))
    assert np.array_equal(out, x)


def test_shift_time_edge_ramp():
    # ramp per time step so shifts are unambiguous
    a = np.arange(L, dtype=np.float32)[None, :, None].repeat(2, axis=0)  # (2,L,1)
    fwd = _shift_time_edge(a, 3)  # delay by 3, head padded with a[0]=0
    assert np.allclose(fwd[0, :3, 0], 0.0)
    assert np.allclose(fwd[0, 3:, 0], np.arange(L - 3))
    back = _shift_time_edge(a, -2)  # advance by 2, tail padded with a[-1]=L-1
    assert np.allclose(back[0, : L - 2, 0], np.arange(2, L))
    assert np.allclose(back[0, L - 2 :, 0], L - 1)


def test_jitter_clean_identity_and_shape():
    x = _windows(seed=11)
    lay = channel_layout(_feature_names())
    out0 = apply_sync_jitter(x, 0, lay, make_rng(1, 0, "jitter", 0, 0))
    assert np.array_equal(out0, x)
    out = apply_sync_jitter(x, 4, lay, make_rng(1, 0, "jitter", 4, 0))
    assert out.shape == x.shape
    # with a non-zero max, at least one relay shifts -> array changes (very likely)
    assert not np.array_equal(out, x)


def test_jitter_deterministic():
    x = _windows(seed=13)
    lay = channel_layout(_feature_names())
    a = apply_sync_jitter(x, 2, lay, make_rng(9, 3, "jitter", 2, 0))
    b = apply_sync_jitter(x, 2, lay, make_rng(9, 3, "jitter", 2, 0))
    assert np.array_equal(a, b)


def test_operators_do_not_mutate_input():
    x = _windows(seed=17)
    x0 = x.copy()
    lay = channel_layout(_feature_names())
    add_gaussian_snr(x, 20, make_rng(1, 0, "noise", 20, 0))
    apply_ct_saturation(x, 0.5, lay, make_rng(1, 0, "ct_saturation", 0.5, 0))
    apply_sync_jitter(x, 4, lay, make_rng(1, 0, "jitter", 4, 0))
    assert np.array_equal(x, x0)
