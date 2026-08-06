"""Unit tests for ``fcl_psp.models.run_model.select_relay_features``.

Operates on synthetic ``(N, L, F)`` arrays. The function is duck-typed on the
config object, so a lightweight ``SimpleNamespace`` stands in for ``MainConfig``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from fcl_psp.models.run_model import select_relay_features

N, L, N_RELAYS, FEAT_PER_RELAY = 4, 10, 8, 3
F = N_RELAYS * FEAT_PER_RELAY


def _windows():
    # Distinct values per feature column so we can verify which columns survive.
    base = np.arange(F, dtype=np.float32)
    return np.broadcast_to(base, (N, L, F)).copy()


def _cfg(**ablation):
    if ablation:
        fields = dict(
            enabled=True,
            n_relays=N_RELAYS,
            features_per_relay=FEAT_PER_RELAY,
            relay_index=None,
            relay_indices=[],
        )
        fields.update(ablation)
        ab = SimpleNamespace(**fields)
    else:
        ab = SimpleNamespace(enabled=False)
    return SimpleNamespace(ablation=ab)


def test_full_mode_returns_all_features():
    w = _windows()
    out, meta = select_relay_features(w, _cfg())
    assert out.shape == (N, L, F)
    assert meta["mode"] == "full"


def test_single_relay_selects_three_columns():
    w = _windows()
    out, meta = select_relay_features(w, _cfg(mode="single_relay", relay_index=2))
    assert out.shape == (N, L, FEAT_PER_RELAY)
    # relay 2 -> columns [6, 7, 8]
    np.testing.assert_array_equal(out[0, 0], [6, 7, 8])
    assert meta["selected_relays"] == [2]


def test_relay_subset_dedupes_and_sorts():
    w = _windows()
    out, meta = select_relay_features(w, _cfg(mode="relay_subset", relay_indices=[3, 1, 1]))
    assert out.shape == (N, L, 2 * FEAT_PER_RELAY)
    assert meta["selected_relays"] == [1, 3]


def test_drop_one_relay_removes_three_columns():
    w = _windows()
    out, _ = select_relay_features(w, _cfg(mode="drop_one_relay", relay_index=0))
    assert out.shape == (N, L, F - FEAT_PER_RELAY)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        select_relay_features(_windows(), _cfg(mode="bogus"))


def test_single_relay_requires_index():
    with pytest.raises(ValueError):
        select_relay_features(_windows(), _cfg(mode="single_relay", relay_index=None))
