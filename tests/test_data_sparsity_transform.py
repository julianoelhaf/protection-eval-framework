"""Unit tests for ``fcl_psp.models.data_sparsity.apply_sparsity_transform``.

This is the hot path of every run: it flattens (N, L, F) windows to (N, L*F) and
optionally applies sensor-availability degradation. The transform must

1. produce the time-major flattening the manuscript specifies,
2. never modify its input, and
3. skip copying when no degradation is configured (the default path), because the
   input is typically a multi-GB memmap.

The equivalence test pins (1) against an explicit reference flatten, so the
no-copy fast path cannot silently change the values any model is trained on.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fcl_psp.models.data_sparsity import apply_sparsity_transform, sparsity_is_disabled

N, L, F = 6, 4, 12  # 2 relays x 6 channels, small but shaped like the real thing


def _cfg(**overrides):
    """Config with every degradation at its documented 'disabled' sentinel."""
    ds = dict(
        bus_failure_id=0,
        current_loss=False,
        downsampling_factor=1,
        phase_failure_id="None",
        relay_failure_ids=[0],
        voltage_loss=False,
        zeroing_duration_s=0.0,
    )
    ds.update(overrides)
    return SimpleNamespace(
        data_sparsity=SimpleNamespace(**ds),
        dataset=SimpleNamespace(sampling_frequency=6400, topology="test"),
        ablation=SimpleNamespace(enabled=False, features_per_relay=6, n_relays=2),
    )


def _windows(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, L, F)).astype(np.float32)


def test_default_config_is_detected_as_disabled():
    assert sparsity_is_disabled(_cfg()) is True


@pytest.mark.parametrize(
    "override",
    [
        {"voltage_loss": True},
        {"current_loss": True},
        {"bus_failure_id": 1},
        {"phase_failure_id": "A"},
        {"zeroing_duration_s": 0.005},
        {"downsampling_factor": 2},
        {"relay_failure_ids": [1]},
    ],
)
def test_any_enabled_option_disables_the_fast_path(override):
    assert sparsity_is_disabled(_cfg(**override)) is False


def test_flatten_is_time_major_and_value_preserving():
    """Disabled path must equal an explicit reference flatten, exactly."""
    w = _windows()
    out, shape = apply_sparsity_transform(w, _cfg())

    assert out.shape == (N, L * F)
    assert shape == (N, L, F)
    # Time-major: all features of one timestep stay together, then the next timestep.
    np.testing.assert_array_equal(out, w.reshape(N, L * F))
    for i in range(N):
        for t in range(L):
            np.testing.assert_array_equal(out[i, t * F : (t + 1) * F], w[i, t, :])


def test_input_is_not_modified_on_the_disabled_path():
    w = _windows()
    before = w.copy()
    apply_sparsity_transform(w, _cfg())
    np.testing.assert_array_equal(w, before)


def test_input_is_not_modified_when_masking_is_enabled():
    """The masking helpers write in place, so the transform must copy first."""
    w = _windows()
    before = w.copy()
    out, _ = apply_sparsity_transform(w, _cfg(voltage_loss=True))
    np.testing.assert_array_equal(w, before, err_msg="input array was mutated")
    # ...and the mask actually did something.
    assert np.count_nonzero(out) < np.count_nonzero(before)


def test_disabled_path_avoids_copying():
    """A view (not a copy) is returned, keeping peak memory at one array.

    Guards the memory fix: three full copies of a ~5 GB FC array used to be made
    before any model saw the data.
    """
    w = _windows()
    out, _ = apply_sparsity_transform(w, _cfg())
    assert out.base is not None, "expected a reshaped view, not a fresh allocation"
    assert np.shares_memory(out, w)


def test_memmap_input_is_not_written_through(tmp_path):
    """Reshaping a memmap yields a view; callers must still not affect the cache file."""
    path = tmp_path / "X.raw"
    src = _windows(seed=1)
    src.tofile(path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=(N, L, F))

    out, _ = apply_sparsity_transform(mm, _cfg())
    np.testing.assert_array_equal(out, src.reshape(N, L * F))

    # Fancy indexing (what every caller does) copies, so downstream in-place work
    # such as StandardScaler(copy=False) cannot reach the file.
    rows = out[np.array([0, 2, 4])]
    assert not np.shares_memory(rows, mm)
    rows[:] = 0.0
    del mm
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.float32).reshape(N, L, F), src)
