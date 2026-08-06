"""Leakage-boundary tests for the corrected MOMENT fold-local preprocessing.

Dataset-free: synthetic numpy arrays with known ground truth, mirroring the style of
``tests/test_relay_features.py``. These tests must NOT require ``momentfm``, ``torch`` or a GPU --
they exercise only the pure helpers in ``fcl_psp.models.moment_cv_utils`` and the aggregator, so
they run in the light ``.venv`` (numpy + scikit-learn only).
"""

import importlib.util
import os
import re

import numpy as np
import pytest

from fcl_psp.models.moment_cv_utils import (
    fit_flat_window_scaler,
    fit_probe,
    make_folds,
    probe_sklearn,
    sample_probe_rows,
)

# The aggregator lives under runs/harnesses (not an installed package); load it by path.
_AGG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "harnesses",
    "aggregate_moment_results.py",
)
_spec = importlib.util.spec_from_file_location("aggregate_moment_results", _AGG_PATH)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

# The line format the corrected aggregator must keep producing (copied from docs/gen_site_data.py).
MOMENT_RE = re.compile(
    r"^\[(?P<tag>moment_\w+)\] \S+ moment W=[\d.]+ mode=(?P<mode>probe|head) "
    r"(?P<metric>macroF1|MAE) mean=(?P<mean>[\d.]+) std=(?P<std>[\d.]+) per_fold=\[(?P<folds>[^\]]*)\]"
)

SEED = 42


def _windows(n=60, L=4, F=3):
    return np.zeros((n, L, F), dtype=np.float32)


# --- Test A: the raw-window scaler sees training rows only ---------------------------------------


def test_A_scaler_ignores_test_rows():
    windows = _windows()
    n, L, F = windows.shape
    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, n)
    rng = np.random.RandomState(SEED)
    # Training windows centred near zero; test windows carry an extreme sentinel.
    windows[train_idx] = rng.normal(0.0, 1.0, size=(len(train_idx), L, F)).astype(np.float32)
    windows[test_idx] = 1e6

    scaler = fit_flat_window_scaler(windows, train_idx, chunk_size=7)

    flat_train = windows[train_idx].reshape(len(train_idx), L * F)
    assert np.allclose(scaler.mean_, flat_train.mean(axis=0), atol=1e-4)
    # The 1e6 test sentinel never entered the fit.
    assert np.abs(scaler.mean_).max() < 100.0


# --- Test B: scaler fit count equals the number of training windows ------------------------------


@pytest.mark.parametrize("chunk_size", [1, 7, 40, 4096])
def test_B_scaler_fit_count(chunk_size):
    windows = _windows(n=50)
    train_idx = np.arange(0, 37)
    scaler = fit_flat_window_scaler(windows, train_idx, chunk_size=chunk_size)
    assert int(scaler.n_samples_seen_) == len(train_idx)


# --- Test C: episode groups are disjoint across every fold ---------------------------------------


def test_C_group_folds_disjoint():
    n_groups = 25
    per_group = 4
    sample_ids = np.repeat(np.arange(n_groups), per_group)
    y = (np.arange(len(sample_ids)) % 3).astype(np.int64)
    folds = make_folds(y, sample_ids, n_splits=5)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        train_groups = set(sample_ids[train_idx].tolist())
        test_groups = set(sample_ids[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups)
        assert len(test_idx) > 0


# --- Test D: the probe fits scaler + estimator on the training array only ------------------------


def test_D_probe_fits_on_train_only():
    rng = np.random.RandomState(SEED)
    E_train = rng.normal(0.0, 1.0, size=(200, 8)).astype(np.float32)
    y_train = (E_train[:, 0] > 0).astype(np.int64)
    E_test = rng.normal(0.0, 1.0, size=(20, 8)).astype(np.float32)
    E_test_extreme = E_test.copy()
    E_test_extreme[0, :] = 1e6  # only the first test row is contaminated

    scaler, est, n_fit = fit_probe(E_train, y_train, "clf", seed=SEED, cap=1000)
    sub = sample_probe_rows(len(E_train), 1000, SEED)
    # Fitted embedding scaler equals the training-subset statistics, untouched by any test data.
    assert np.allclose(scaler.mean_, E_train[sub].mean(axis=0), atol=1e-4)
    assert np.abs(scaler.mean_).max() < 100.0
    assert n_fit == len(sub)

    # The fit is independent of E_test contents: predictions on the shared (rows 1..) block match,
    # regardless of the extreme sentinel placed only in row 0.
    p_benign, n1 = probe_sklearn(E_train, y_train, E_test, "clf", SEED, cap=1000)
    p_extreme, n2 = probe_sklearn(E_train, y_train, E_test_extreme, "clf", SEED, cap=1000)
    assert n1 == n2 == n_fit
    assert np.array_equal(p_benign[1:], p_extreme[1:])


# --- Test E: aggregation rejects incomplete results in strict mode -------------------------------


def _rec(tag, fold, target, task, wl, ntrain=800, ntest=200):
    return {
        "tag": tag,
        "outer_fold": fold,
        "target": target,
        "task": task,
        "window_length": wl,
        "n_train_windows": ntrain,
        "n_test_windows": ntest,
        "n_train_groups": 40,
        "n_test_groups": 10,
        "group_overlap": 0,
        "raw_scaler_fit_windows": ntrain,
        "probe_fit_windows": min(ntrain, 40000),
        "probe_score": 0.9,
        "head_score": 0.95,
        "head_epochs": 30,
        "head_early_stopped": True,
        "smoke": False,
    }


_CONFIGS = [
    ("moment_fc_W20", "event_type", "clf", 0.02),
    ("moment_fc_W50", "event_type", "clf", 0.05),
    ("moment_fl_W20", "y_fault_location", "reg", 0.02),
    ("moment_fl_W50", "y_fault_location", "reg", 0.05),
]


def _full_records():
    return [
        _rec(tag, fold, target, task, wl)
        for (tag, target, task, wl) in _CONFIGS
        for fold in (1, 2, 3, 4, 5)
    ]


def test_E_strict_rejects_missing_fold():
    records = _full_records()
    # Drop fold 5 of the first config -> 19 records, one config has only 4 folds.
    incomplete = [r for r in records if not (r["tag"] == "moment_fc_W20" and r["outer_fold"] == 5)]
    assert len(incomplete) == 19
    with pytest.raises(agg.AggregationError):
        agg.aggregate(incomplete, strict=True)
    # Non-strict must not raise (it warns and proceeds).
    agg.aggregate(incomplete, strict=False)


def test_E_strict_rejects_duplicate_fold():
    records = _full_records()
    dup = records + [_rec("moment_fc_W20", 3, "event_type", "clf", 0.02)]
    with pytest.raises(agg.AggregationError):
        agg.aggregate(dup, strict=True)


def test_E_strict_rejects_group_overlap():
    records = _full_records()
    records[0]["group_overlap"] = 2
    with pytest.raises(agg.AggregationError):
        agg.aggregate(records, strict=True)


def test_E_strict_rejects_scaler_mismatch():
    records = _full_records()
    records[0]["raw_scaler_fit_windows"] = records[0]["n_train_windows"] - 1
    with pytest.raises(agg.AggregationError):
        agg.aggregate(records, strict=True)


def test_aggregate_valid_produces_parser_compatible_lines():
    lines = agg.aggregate(_full_records(), strict=True)
    assert len(lines) == 8
    tags = [MOMENT_RE.match(ln).group("tag") for ln in lines]
    assert tags == agg.FINAL_TAG_ORDER
    for ln in lines:
        m = MOMENT_RE.match(ln)
        assert m is not None, f"line not parser-compatible: {ln}"
        # per_fold must contain exactly five values.
        assert len(m.group("folds").split(",")) == 5
