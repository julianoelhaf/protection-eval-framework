"""Unit tests for ``fcl_psp.models.model_utils``.

Validates the task-type resolution, the model registries, and that
``create_model_from_name`` builds the expected sklearn estimator for a couple of
simple models using a lightweight duck-typed config.
"""

from types import SimpleNamespace

import pytest
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fcl_psp.models.model_utils import (
    CLASSIFIERS,
    REGRESSORS,
    create_model_from_name,
    get_task_type,
)


def _cfg(model_name, target_label):
    return SimpleNamespace(
        model=SimpleNamespace(model_name=model_name, n_jobs=None),
        training=SimpleNamespace(target_label=target_label, random_state=42),
    )


@pytest.mark.parametrize(
    "target_label,expected",
    [
        ("y_fault_present", "binary"),
        ("y_fault_location", "regression"),
        ("event_type", "multiclass"),
        ("y_fault_line", "multiclass"),
    ],
)
def test_get_task_type(target_label, expected):
    assert get_task_type(_cfg("decision_tree_classifier", target_label)) == expected


def test_get_task_type_unknown_raises():
    with pytest.raises(ValueError):
        get_task_type(_cfg("decision_tree_classifier", "not_a_real_target"))


def test_registries_are_estimator_classes():
    assert CLASSIFIERS and REGRESSORS
    for registry in (CLASSIFIERS, REGRESSORS):
        for name, cls in registry.items():
            assert isinstance(cls, type), name
            assert issubclass(cls, BaseEstimator), name


def test_create_classifier():
    model = create_model_from_name(_cfg("decision_tree_classifier", "event_type"))
    assert isinstance(model, DecisionTreeClassifier)


def test_create_regressor():
    model = create_model_from_name(_cfg("decision_tree_regressor", "y_fault_location"))
    assert isinstance(model, DecisionTreeRegressor)


def test_create_unsupported_model_raises():
    with pytest.raises(ValueError):
        create_model_from_name(_cfg("not_a_model", "event_type"))
