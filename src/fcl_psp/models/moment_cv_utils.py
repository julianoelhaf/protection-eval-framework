"""Fold-local cross-validation preprocessing for the MOMENT-1-large baseline.

Pure numpy + scikit-learn helpers (no ``torch`` / ``momentfm``) so the leakage-critical
preprocessing can be unit-tested without a GPU or the deep-learning environment.

The evaluation is episode-grouped five-fold ``GroupKFold``. For a single outer fold every
fitted object -- the raw-window standardizer, the probe's embedding standardizer, and the
probe estimator -- is fit **only** on that fold's training partition. The frozen MOMENT
backbone is applied separately within each outer fold, so the embeddings depend on the
fold-fitted raw-window scaler and are therefore fold-specific (they are *not* reused across
folds). The held-out fold is used only for prediction and scoring.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROBE_CAP_DEFAULT = 40000


def make_folds(y, sample_ids, n_splits=5):
    """Episode-grouped outer folds. Returns a list of ``(train_idx, test_idx)`` pairs.

    Grouping is by ``sample_ids`` (episode IDs), so no episode spans the train/test boundary.
    ``y`` is accepted for API parity with the harness but does not influence the split.
    """
    y = np.asarray(y)
    sample_ids = np.asarray(sample_ids)
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups=sample_ids))


def fit_flat_window_scaler(windows, train_indices, chunk_size=4096):
    """Fit a per-feature ``StandardScaler`` on the flattened raw training windows only.

    ``windows`` has shape ``(N, window_length, n_features)`` (may be a memmap). Each window is
    flattened to ``window_length * n_features`` and the scaler is built incrementally with
    ``partial_fit`` over ``train_indices`` in chunks, so the full training partition is used
    without materializing it all at once. The outer-test indices never enter the fit.
    """
    train_indices = np.asarray(train_indices)
    scaler = StandardScaler()
    _, window_length, n_features = windows.shape
    for start in range(0, len(train_indices), chunk_size):
        idx = train_indices[start : start + chunk_size]
        batch = np.asarray(windows[idx], dtype=np.float32).reshape(
            len(idx), window_length * n_features
        )
        scaler.partial_fit(batch)
    return scaler


def sample_probe_rows(n_rows, cap, seed):
    """Deterministic ``<=cap`` row indices into an embedding matrix of ``n_rows`` rows.

    Uses ``np.random.RandomState(seed)`` so a given (n_rows, cap, seed) always yields the same
    subset. Returns all rows (sorted) when ``n_rows <= cap``.
    """
    if n_rows <= cap:
        return np.arange(n_rows)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_rows, cap, replace=False))


def fit_probe(E_train, y_train, task, seed, cap=PROBE_CAP_DEFAULT):
    """Fit the linear probe on a deterministic subset of the outer-training embeddings.

    Both the embedding ``StandardScaler`` and the estimator are fit only on the sampled
    training rows. Returns ``(scaler, estimator, n_fit_rows)``. There is no test-data argument
    here by construction, so nothing from the outer-test fold can enter the fit.
    """
    y_train = np.asarray(y_train)
    sub = sample_probe_rows(len(E_train), cap, seed)
    Xf = np.asarray(E_train[sub], dtype=np.float32)
    scaler = StandardScaler().fit(Xf)
    Xf = scaler.transform(Xf)
    est = (
        LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
        if task == "clf"
        else Ridge(alpha=1e4, solver="lsqr")
    )
    est.fit(Xf, y_train[sub])
    return scaler, est, int(len(sub))


def probe_predict(scaler, est, E_test, chunk=20000):
    """Apply the train-fitted scaler unchanged to ``E_test`` and predict, in memory-bounded chunks."""
    preds = []
    for i in range(0, len(E_test), chunk):
        Xc = scaler.transform(np.asarray(E_test[i : i + chunk], dtype=np.float32))
        preds.append(est.predict(Xc))
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.float64)


def probe_sklearn(E_train, y_train, E_test, task, seed, cap=PROBE_CAP_DEFAULT):
    """Train/test-separated linear probe.

    Samples ``<=cap`` rows from ``E_train``, fits the embedding scaler and estimator on that
    training subset only, applies the scaler unchanged to ``E_test`` and predicts every
    outer-test row. Returns ``(predictions, n_probe_fit_rows)``.
    """
    scaler, est, n_fit = fit_probe(E_train, y_train, task, seed, cap)
    preds = probe_predict(scaler, est, E_test)
    return preds, n_fit
