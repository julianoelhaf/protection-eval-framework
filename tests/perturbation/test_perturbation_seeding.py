"""Unit tests for ``fcl_psp.perturbation.seeding`` (deterministic RNG derivation)."""

import pytest

from fcl_psp.perturbation.seeding import AXIS_ID, make_rng, seed_entropy


def _first_draw(rng):
    return rng.standard_normal(4).tolist()


def test_same_cell_reproduces():
    a = _first_draw(make_rng(123, 2, "noise", 20, 1))
    b = _first_draw(make_rng(123, 2, "noise", 20, 1))
    assert a == b


@pytest.mark.parametrize(
    "kw",
    [
        dict(fold=3),  # different fold
        dict(rep=2),  # different realization
        dict(level=30),  # different level
        dict(axis="jitter"),  # different axis
        dict(seed_global=999),  # different global seed
    ],
)
def test_different_cells_diverge(kw):
    base = dict(seed_global=123, fold=2, axis="noise", level=20, rep=1)
    a = _first_draw(make_rng(**base))
    other = {**base, **kw}
    b = _first_draw(make_rng(**other))
    assert a != b


def test_clean_sentinels_share_level_id():
    # "clean"/"no_sat"/None all map to the identity level id -> same stream.
    a = _first_draw(make_rng(1, 0, "noise", "clean", 0))
    b = _first_draw(make_rng(1, 0, "noise", None, 0))
    assert a == b


def test_unknown_axis_raises():
    with pytest.raises(ValueError):
        make_rng(1, 0, "bogus_axis", 20, 0)


def test_seed_entropy_shape():
    ent = seed_entropy(1, 0, "noise", 20, 0)
    assert isinstance(ent, list) and len(ent) == 5
    assert ent[1] == AXIS_ID["noise"]
