from typing import Any, Optional, Tuple

import wandb
from psp_helper.config import MainConfig
from psp_helper.constants import TaskType


def wandb_log_dataset_params(
    n_samples: int,
    window_shape: Tuple[int, int, int],
    task_type: TaskType,
    config: MainConfig,
) -> None:
    """
    Log dataset identity, windowing setup, task definition,
    and evaluation protocol. These parameters define the *experimental context*
    and should be constant across model sweeps.
    """
    n_windows, n_timesteps, n_feats_per_step = window_shape

    cfg = {
        # ---- Ablation setup ----
        "ablation/enabled": bool(config.ablation.enabled),
        "ablation/mode": config.ablation.mode,
        "ablation/relay_index": (
            int(config.ablation.relay_index) if config.ablation.relay_index is not None else None
        ),
        "ablation/relay_indices": list(config.ablation.relay_indices),
        "ablation/n_relays": int(config.ablation.n_relays),
        "ablation/features_per_relay": int(config.ablation.features_per_relay),
        # ---- Dataset identity ----
        "dataset/topology": config.dataset.topology,
        "dataset/sampling_frequency": int(config.dataset.sampling_frequency),
        "dataset/nominal_frequency": int(config.dataset.frequency),
        "dataset/n_samples_total": int(n_samples),
        # ---- Window extraction ----
        "window/length_s": float(config.window_extraction.window_length),
        "window/step_s": float(config.window_extraction.step_length_seconds),
        "window/period_of_interest_s": float(config.window_extraction.period_of_interest),
        "window/n_windows": int(n_windows),
        "window/n_timesteps": int(n_timesteps),
        # ---- Feature dimensionality ----
        "features/per_timestep": int(n_feats_per_step),
        "features/flat": int(n_timesteps * n_feats_per_step),
        # ---- Task definition ----
        "task/type": task_type,
        "task/target_label": config.training.target_label,
        # ---- Evaluation protocol ----
        "cv/n_splits": int(config.training.n_splits),
        "cv/random_state": int(config.training.random_state),
        # ---- Data sparsity / corruption ----
        "sparsity/current_loss": bool(config.data_sparsity.current_loss),
        "sparsity/voltage_loss": bool(config.data_sparsity.voltage_loss),
        "sparsity/downsampling_factor": int(config.data_sparsity.downsampling_factor),
        "sparsity/zeroing_duration_s": float(config.data_sparsity.zeroing_duration_s),
        "sparsity/bus_failure_id": int(config.data_sparsity.bus_failure_id),
        "sparsity/relay_failure_ids": list(config.data_sparsity.relay_failure_ids),
        "sparsity/phase_failure_id": str(config.data_sparsity.phase_failure_id),
    }

    wandb.config.update(cfg, allow_val_change=True)


def _as_optional_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"none", "null", ""}:
            return None
        return int(s)
    return int(x)


def wandb_log_model_params(config: MainConfig) -> None:
    """
    Log model-specific hyperparameters.
    This function is intentionally selective to keep W&B clean.
    """
    cfg = {
        "model/name": config.model.model_name,
    }

    # ---- MLP ----
    if config.model.model_name in ("mlp_classifier", "mlp_regressor"):
        hls = list(config.model.mlp.hidden_layer_sizes)

        cfg.update(
            {
                "mlp/activation": config.model.mlp.activation,
                "mlp/alpha": float(config.model.mlp.alpha),
                "mlp/batch_size": config.model.mlp.batch_size,
                "mlp/early_stopping": bool(config.model.mlp.early_stopping),
                "mlp/hidden_layer_sizes": hls,
                "mlp/learning_rate_init": float(config.model.mlp.learning_rate_init),
                "mlp/max_iter": config.model.mlp.max_iter,
                "mlp/n_iter_no_change": int(config.model.mlp.n_iter_no_change),
            }
        )

    # ---- HistGradientBoosting ----
    if config.model.model_name in (
        "hist_gradient_boosting_classifier",
        "hist_gradient_boosting_regressor",
    ):
        max_depth = _as_optional_int(config.model.hgb.max_depth)
        cfg.update(
            {
                "hgb/is_depth_limited": config.model.hgb.max_depth is not None,
                "hgb/l2_regularization": float(config.model.hgb.l2_regularization),
                "hgb/learning_rate": float(config.model.hgb.learning_rate),
                "hgb/max_depth": max_depth,
                "hgb/max_iter": int(config.model.hgb.max_iter),
                "hgb/min_samples_leaf": int(config.model.hgb.min_samples_leaf),
            }
        )

    wandb.config.update(cfg, allow_val_change=True)
