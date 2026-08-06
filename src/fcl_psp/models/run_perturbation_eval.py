#!/usr/bin/env python3
"""Experiment B driver — measurement-fidelity degradation (train-clean / test-perturbed).

Trains the clean per-fold models ONCE (reusing the framework's load -> onset-filter
-> reshape -> GroupKFold -> per-fold-scaler path), caches (clean scaler, clean model,
clean test windows), then inner-loops ``(axis, level, realization)`` as inference
only: perturb a COPY of the test windows, apply the CLEAN-fit scaler, predict.

Invariants preserved (see agent/coding_plan.md §5.6):
  * scaler fit on CLEAN train only; perturbation touches test copies after the split;
  * GroupKFold split identical to run_model.py (same folds);
  * clean level of every axis is the identity operator -> reproduces the reference.

The clean numeric path in run_model.py is NOT modified.

Example (FL, 20 ms):
  python -m fcl_psp.models.run_perturbation_eval \\
      training.target_label=y_fault_location model.model_name=mlp_regressor \\
      window_extraction.window_length=0.020 \\
      window_extraction.windows_local_dir=/path/to/windows \\
      tracking.mode=disabled +perturbation.axes=[noise,ct_saturation,jitter]
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from psp_helper.config import MainConfig
from psp_helper.windows_helper import load_meta_data, load_windows_and_labels
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from fcl_psp.models import data_sparsity
from fcl_psp.models.model_utils import create_model_from_name, get_task_type
from fcl_psp.models.run_model import (
    get_sample_ids_and_fault_targets,
    select_relay_features,
    write_filtered_memmap,
)
from fcl_psp.perturbation.config import PerturbationConfig
from fcl_psp.perturbation.operators import _is_clean, apply_axis, channel_layout
from fcl_psp.perturbation.seeding import make_rng, seed_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_perturbation_eval")


def _load_perturbation_cfg(hydra_cfg: MainConfig) -> PerturbationConfig:
    base = OmegaConf.structured(PerturbationConfig)
    yaml_path = Path(__file__).resolve().parents[3] / "config" / "perturbation" / "default.yaml"
    if yaml_path.exists():
        base = OmegaConf.merge(base, OmegaConf.load(yaml_path))
    override = OmegaConf.select(hydra_cfg, "perturbation")
    if override is not None:
        base = OmegaConf.merge(base, override)
    return OmegaConf.to_object(base)  # type: ignore[return-value]


def _is_identity(axis: str, level) -> bool:
    if axis == "jitter":
        return level is None or int(level) == 0
    return _is_clean(level)


def _metric(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if task_type in ("binary", "multiclass"):
        return {
            "metric": "macro_f1",
            "value": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    return {"metric": "mae", "value": float(mean_absolute_error(y_true, y_pred))}


@hydra.main(version_base=None, config_path="../../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:
    pcfg = _load_perturbation_cfg(config)
    task_type = get_task_type(config)
    window_ms = round(config.window_extraction.window_length * 1000)

    # ---- load (framework path) ----
    windows, labels, row_indices = load_windows_and_labels(config)
    if row_indices is not None:
        fpath = (
            Path(config.window_extraction.windows_local_dir)
            / f"X_fault_only_{config.dataset.topology}_W{str(config.window_extraction.window_length).replace('.', 'p')}.raw"
        )
        windows = write_filtered_memmap(windows, row_indices, fpath)
    windows, _ = select_relay_features(windows, config)
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, config)
    window_data, new_shape = data_sparsity.apply_sparsity_transform(windows, config)
    _, Ln, Fn = new_shape
    feature_names = load_meta_data(config)["feature_names"]
    layout = channel_layout(feature_names)
    model = create_model_from_name(config)
    gkf = GroupKFold(n_splits=int(config.training.n_splits))
    logger.info(
        "Loaded window_data=%s task=%s W=%dms model=%s",
        window_data.shape,
        task_type,
        window_ms,
        config.model.model_name,
    )

    # ---- Phase 1: train clean fold models once, cache ----
    folds = []
    for i, (tr, te) in enumerate(gkf.split(window_data, y, groups=sample_ids), start=1):
        scaler = StandardScaler(copy=True)  # copy=True so transform never mutates cached test
        X_train = scaler.fit_transform(window_data[tr])
        model_fold = copy.deepcopy(model).fit(X_train, y[tr])
        del X_train
        folds.append(
            {
                "fold": i,
                "scaler": scaler,
                "model": model_fold,
                "Xte": np.asarray(window_data[te]).copy(),  # (N_te, L*F) writable clean copy
                "yte": np.asarray(y[te]),
            }
        )
        logger.info("Clean fold %d trained (n_train=%d n_test=%d)", i, len(tr), len(te))

    # ---- Phase 2: perturbation sweep (inference only) ----
    rows: List[dict] = []
    for f in folds:
        Xte3 = f["Xte"].reshape(-1, Ln, Fn)  # (N_te, L, F)
        for axis in pcfg.axes:
            for level in pcfg.levels_for(axis):
                reps = 1 if _is_identity(axis, level) else int(pcfg.n_realizations)
                for rep in range(reps):
                    rng = make_rng(pcfg.seed_global, f["fold"], axis, level, rep)
                    Xp3 = apply_axis(axis, level, Xte3.copy(), layout, rng)
                    Xp = f["scaler"].transform(Xp3.reshape(f["Xte"].shape))
                    pred = f["model"].predict(Xp)
                    m = _metric(task_type, f["yte"], pred)
                    rows.append(
                        {
                            "task": task_type,
                            "target": config.training.target_label,
                            "model": config.model.model_name,
                            "window_ms": window_ms,
                            "training_regime": "clean",
                            "axis": axis,
                            "level": str(level),
                            "realization": rep,
                            "fold": f["fold"],
                            "metric": m["metric"],
                            "value": m["value"],
                            "identity": _is_identity(axis, level),
                        }
                    )
        logger.info("Fold %d perturbation sweep done", f["fold"])

    # ---- write outputs ----
    out_dir = (
        Path(pcfg.out_dir)
        / f"{config.model.model_name}_{config.training.target_label}_W{window_ms}ms"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "perturbation_metrics.parquet", index=False)
    # summary: mean/std over folds x realizations per (axis, level)
    summary = (
        df.groupby(["axis", "level", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["axis", "level"])
    )
    summary.to_csv(out_dir / "perturbation_summary.csv", index=False)
    meta = {
        "task": task_type,
        "model": config.model.model_name,
        "window_ms": window_ms,
        "seed_global": pcfg.seed_global,
        "n_realizations": pcfg.n_realizations,
        "axes": list(pcfg.axes),
        "n_splits": int(config.training.n_splits),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed_example": seed_entropy(pcfg.seed_global, 1, "noise", 20, 0),
        "config_resolved": OmegaConf.to_container(
            OmegaConf.create(OmegaConf.to_yaml(config)), resolve=False
        ),
    }
    (out_dir / "perturbation_run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Wrote %s", out_dir / "perturbation_summary.csv")
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
