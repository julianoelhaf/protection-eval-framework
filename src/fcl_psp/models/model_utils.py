import inspect
import logging
from typing import Any, Dict, Type

from psp_helper.config import MainConfig
from psp_helper.constants import FAULT_TARGET_TASK_TYPE, TaskType

# Regression models
# Classification models
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Classifiers
CLASSIFIERS = {
    "ada_boost_classifier": AdaBoostClassifier,
    "bagging_classifier": BaggingClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    "extra_trees_classifier": ExtraTreesClassifier,
    "hist_gradient_boosting_classifier": HistGradientBoostingClassifier,
    "k_neighbors_classifier": KNeighborsClassifier,
    "linear_svc": LinearSVC,
    "logistic_regression": LogisticRegression,
    "mlp_classifier": MLPClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "ridge_classifier": RidgeClassifier,
    "sgd_classifier": SGDClassifier,
    "stacking_classifier": StackingClassifier,
    "svc": SVC,
    "voting_classifier": VotingClassifier,
}

# Regressors
REGRESSORS = {
    "ada_boost_regressor": AdaBoostRegressor,
    "bagging_regressor": BaggingRegressor,
    "decision_tree_regressor": DecisionTreeRegressor,
    "extra_trees_regressor": ExtraTreesRegressor,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
    "k_neighbors_regressor": KNeighborsRegressor,
    "linear_regression": LinearRegression,
    "linear_svr": LinearSVR,
    "mlp_regressor": MLPRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "ridge_regressor": Ridge,
    "sgd_regressor": SGDRegressor,
    "stacking_regressor": StackingRegressor,
    "svr": SVR,
    "voting_regressor": VotingRegressor,
}

LIST_OF_REGRESSORS_WITHOUT_RANDOM_STATE = [
    "stacking_regressor",
    "voting_regressor",
    "svr",
]

LIST_OF_CLASSIFIERS_WITHOUT_RANDOM_STATE = [
    "stacking_classifier",
    "voting_classifier",
    "svc",
]


def get_task_type(config: MainConfig) -> TaskType:
    task_type = FAULT_TARGET_TASK_TYPE.get(config.training.target_label)
    if task_type is None:
        raise ValueError(
            f"Unknown fault_target '{config.training.target_label}' - cannot determine task type."
        )
    return task_type


def _filter_kwargs_for_estimator(
    estimator_cls: Type[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    sig = inspect.signature(estimator_cls.__init__)
    allowed = set(sig.parameters.keys())
    # drop "self"
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _common_kwargs(
    config: MainConfig, model_name: str, estimator_cls: Type[Any]
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}

    # random_state
    if hasattr(config.training, "random_state"):
        if (model_name in LIST_OF_CLASSIFIERS_WITHOUT_RANDOM_STATE) or (
            model_name in LIST_OF_REGRESSORS_WITHOUT_RANDOM_STATE
        ):
            pass
        else:
            kwargs["random_state"] = config.training.random_state

    # n_jobs (only if user set it)
    if getattr(config.model, "n_jobs", None):
        kwargs["n_jobs"] = config.model.n_jobs

    # only keep kwargs that estimator actually accepts
    return _filter_kwargs_for_estimator(estimator_cls, kwargs)


def _build_mlp_classifier(config: MainConfig) -> MLPClassifier:
    mlp = config.model.mlp
    return MLPClassifier(
        hidden_layer_sizes=tuple(getattr(mlp, "hidden_layer_sizes", [100])),
        activation=getattr(mlp, "activation", "relu"),
        alpha=getattr(mlp, "alpha", 1e-4),
        learning_rate_init=getattr(mlp, "learning_rate_init", 0.001),
        batch_size=getattr(mlp, "batch_size", "auto"),
        max_iter=getattr(mlp, "max_iter", 200),
        early_stopping=getattr(mlp, "early_stopping", False),
        n_iter_no_change=getattr(mlp, "n_iter_no_change", 10),
        random_state=getattr(config.training, "random_state", 42),
    )


def _build_mlp_regressor(config: MainConfig) -> MLPRegressor:
    mlp = config.model.mlp
    return MLPRegressor(
        hidden_layer_sizes=tuple(getattr(mlp, "hidden_layer_sizes", [100])),
        activation=getattr(mlp, "activation", "relu"),
        alpha=getattr(mlp, "alpha", 1e-4),
        learning_rate_init=getattr(mlp, "learning_rate_init", 0.001),
        batch_size=getattr(mlp, "batch_size", "auto"),
        max_iter=getattr(mlp, "max_iter", 200),
        early_stopping=getattr(mlp, "early_stopping", False),
        n_iter_no_change=getattr(mlp, "n_iter_no_change", 10),
        random_state=getattr(config.training, "random_state", 42),
    )


def _build_hgb_classifier(config: MainConfig) -> HistGradientBoostingClassifier:
    hgb = config.model.hgb
    return HistGradientBoostingClassifier(
        max_depth=getattr(hgb, "max_depth", None),
        max_iter=getattr(hgb, "max_iter", 100),
        learning_rate=getattr(hgb, "learning_rate", 0.1),
        min_samples_leaf=getattr(hgb, "min_samples_leaf", 20),
        l2_regularization=getattr(hgb, "l2_regularization", 0),
        random_state=getattr(config.training, "random_state", 42),
    )


def _build_hgb_regressor(config: MainConfig) -> HistGradientBoostingRegressor:
    hgb = config.model.hgb
    return HistGradientBoostingRegressor(
        max_depth=getattr(hgb, "max_depth", None),
        max_iter=getattr(hgb, "max_iter", 100),
        learning_rate=getattr(hgb, "learning_rate", 0.1),
        min_samples_leaf=getattr(hgb, "min_samples_leaf", 20),
        l2_regularization=getattr(hgb, "l2_regularization", 0),
        random_state=getattr(config.training, "random_state", 42),
    )


def create_model_from_name(config: MainConfig):
    model_name = config.model.model_name.lower()
    task_type = get_task_type(config)

    if task_type in ("binary", "multiclass"):
        model_class = CLASSIFIERS.get(model_name)
        common_kwargs = _common_kwargs(config, model_name, model_class)
        if model_class is None:
            raise ValueError(f"Unsupported classifier '{model_name}'.")

        # ----- Phase-2 explicit models -----
        if model_name == "mlp_classifier":
            return _build_mlp_classifier(config)

        if model_name == "hist_gradient_boosting_classifier":
            # IMPORTANT: HGB doesn't support n_jobs; remove if present
            common_kwargs.pop("n_jobs", None)
            return _build_hgb_classifier(config)

        # ----- Legacy explicit branches -----
        if model_name in ["random_forest_classifier", "extra_trees_classifier"]:
            kwargs = dict(
                n_estimators=getattr(config.model, "n_estimators", 100),
                max_depth=getattr(config.model, "max_depth", 10),
                **common_kwargs,
            )
            # optional hyperparams
            for k in ["min_samples_split", "min_samples_leaf", "max_features"]:
                v = getattr(config.model, k, None)
                if v is not None:
                    kwargs[k] = v
            return model_class(**kwargs)

        if model_name == "ada_boost_classifier":
            return AdaBoostClassifier(
                n_estimators=getattr(config.model, "n_estimators", 50),
                learning_rate=getattr(config.model, "learning_rate", 1.0),
                random_state=getattr(config.training, "random_state", 42),
            )

        if model_name == "k_neighbors_classifier":
            # sklearn KNN supports n_jobs
            return KNeighborsClassifier(n_jobs=-1)

        if model_name == "logistic_regression":
            # solver-dependent n_jobs; keep explicit and stable
            return LogisticRegression(
                n_jobs=-1,
                random_state=getattr(config.training, "random_state", 42),
                penalty=getattr(config.model, "penalty", "l2"),
                C=getattr(config.model, "C", 1.0),
                solver=getattr(config.model, "solver", "lbfgs"),
            )

        if model_name == "stacking_classifier":
            return StackingClassifier(
                estimators=[
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            random_state=getattr(config.training, "random_state", 42)
                        ),
                    ),
                    (
                        "mlp",
                        MLPClassifier(random_state=getattr(config.training, "random_state", 42)),
                    ),
                ],
                final_estimator=LogisticRegression(),
            )

        if model_name == "voting_classifier":
            return VotingClassifier(
                estimators=[
                    (
                        "hgb",
                        HistGradientBoostingClassifier(
                            random_state=getattr(config.training, "random_state", 42)
                        ),
                    ),
                    (
                        "mlp",
                        MLPClassifier(random_state=getattr(config.training, "random_state", 42)),
                    ),
                ],
            )

        # default
        return model_class(**common_kwargs)

    if task_type == "regression":
        model_class = REGRESSORS.get(model_name)
        common_kwargs = _common_kwargs(config, model_name, model_class)
        if model_class is None:
            raise ValueError(f"Unsupported regressor '{model_name}'.")

        # ----- Phase-2 explicit models -----
        if model_name == "mlp_regressor":
            return _build_mlp_regressor(config)

        if model_name == "hist_gradient_boosting_regressor":
            common_kwargs.pop("n_jobs", None)
            return _build_hgb_regressor(config)

        # ----- Legacy explicit branches -----
        if model_name in ["random_forest_regressor", "extra_trees_regressor"]:
            return model_class(
                max_depth=getattr(config.model, "max_depth", 10),
                n_estimators=getattr(config.model, "n_estimators", 100),
                **common_kwargs,
            )

        if model_name == "ada_boost_regressor":
            return AdaBoostRegressor(
                n_estimators=getattr(config.model, "n_estimators", 50),
                learning_rate=getattr(config.model, "learning_rate", 1.0),
                random_state=getattr(config.training, "random_state", 42),
            )

        if model_name == "linear_svr":
            return LinearSVR(
                max_iter=2000,
                random_state=getattr(config.training, "random_state", 42),
            )

        if model_name == "ridge_regressor":
            return Ridge(
                alpha=getattr(config.model, "alpha", 1.0),
                random_state=getattr(config.training, "random_state", 42),
            )

        if model_name == "k_neighbors_regressor":
            return KNeighborsRegressor(n_jobs=-1)

        if model_name == "linear_regression":
            # LinearRegression supports n_jobs in sklearn
            return LinearRegression(n_jobs=-1)

        if model_name == "stacking_regressor":
            estimators = [
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        random_state=getattr(config.training, "random_state", 42)
                    ),
                ),
                ("mlp", MLPRegressor(random_state=getattr(config.training, "random_state", 42))),
            ]
            # StackingRegressor itself doesn't take random_state; don't pass common_kwargs blindly
            common_kwargs.pop("random_state", None)
            common_kwargs.pop("n_jobs", None)
            return StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression(),
                **common_kwargs,
            )

        if model_name == "voting_regressor":
            estimators = [
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        random_state=getattr(config.training, "random_state", 42)
                    ),
                ),
                ("mlp", MLPRegressor(random_state=getattr(config.training, "random_state", 42))),
            ]
            common_kwargs.pop("random_state", None)
            common_kwargs.pop("n_jobs", None)
            return VotingRegressor(
                estimators=estimators,
                **common_kwargs,
            )

        return model_class(**common_kwargs)

    raise ValueError(f"Unsupported task type '{task_type}'.")
