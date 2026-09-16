#!/usr/bin/env python3
"""Experiment A driver — conventional protection baselines under the framework protocol.

Reuses the framework data path (``load_windows_and_labels`` + ``load_meta_data``),
the same deterministic episode-grouped ``GroupKFold`` split, and the same metrics
(macro-F1 for FC, MAE %line for FL) as the ML models, so results sit in the same
tables. Conventional relays operate on RAW physical volts/amps — NO StandardScaler.

Dispatch on ``+baseline.task``:
  * ``fc``            — symmetrical-component phase selector; tau_p/tau_g fit on
                        train folds only. Requires ``training.target_label=event_type``.
  * ``fl_two_ended``  — synchronized two-ended positive-sequence FL (relay-pair).
  * ``fl_one_ended``  — one-ended reactance FL (single-relay).
                        Both require ``training.target_label=y_fault_location``.

Example:
  python -m fcl_psp.baselines.run_conventional_baselines \\
      training.target_label=y_fault_location \\
      window_extraction.window_length=0.020 \\
      window_extraction.windows_local_dir=/path/to/windows \\
      +baseline.task=fl_two_ended tracking.mode=disabled
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
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

from fcl_psp.baselines.data.line_registry import LineRegistry
from fcl_psp.baselines.estimators import (
    ConventionalFaultClassifier,
    SingleEndedFaultLocator,
    TwoEndedFaultLocator,
    faulted_phase_current_mag,
)
from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_conventional_baselines")


@dataclass
class _Phasor:
    samples_per_cycle: int = 128
    cycle: str = "last"
    scaling: str = "peak"


@dataclass
class _Fc:
    pickup_grid: List[float] = field(default_factory=lambda: [1.2, 1.5, 2.0, 3.0])
    ground_grid: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])


@dataclass
class _Fl:
    current_sign: int = 1
    min_current_ratio: float = 0.01
    neglect_mutual: bool = True
    terminal: str = "S"


@dataclass
class BaselineConfig:
    task: str = "fl_two_ended"
    observability: str = "relay_pair"
    labels_csv_path: str = "/path/to/datasets/PROTECT-90/hv_double_line_90kv_labels.csv"
    samples_per_cycle: int = 128
    cycle: str = "last"
    scaling: str = "peak"
    fc: _Fc = field(default_factory=_Fc)
    fl: _Fl = field(default_factory=_Fl)
    out_dir: str = "reports/baselines"


def _load_baseline_cfg(hydra_cfg: MainConfig) -> BaselineConfig:
    base = OmegaConf.structured(BaselineConfig)
    yaml_path = Path(__file__).resolve().parents[3] / "config" / "baseline" / "default.yaml"
    if yaml_path.exists():
        base = OmegaConf.merge(base, OmegaConf.load(yaml_path))
    override = OmegaConf.select(hydra_cfg, "baseline")
    if override is not None:
        base = OmegaConf.merge(base, override)
    return OmegaConf.to_object(base)  # type: ignore[return-value]


def _fold_stats(values: List[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "per_fold": arr.tolist()}


def _mae_ignore_nan(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    ok = ~np.isnan(y_pred)
    n_valid = int(ok.sum())
    if n_valid == 0:
        return float("nan"), 0
    return float(mean_absolute_error(y_true[ok], y_pred[ok])), n_valid


@hydra.main(version_base=None, config_path="../../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:
    bcfg = _load_baseline_cfg(config)
    spc = int(bcfg.samples_per_cycle)
    fs = int(config.dataset.sampling_frequency)
    L = int(round(config.window_extraction.window_length * fs))
    task = bcfg.task
    window_ms = round(config.window_extraction.window_length * 1000)

    run = {
        "task": task,
        "observability": bcfg.observability,
        "window_ms": window_ms,
        "n_splits": int(config.training.n_splits),
        "samples_per_cycle": spc,
        "L": L,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_cfg": asdict(bcfg),
    }

    out_dir = Path(bcfg.out_dir) / f"{task}_W{window_ms}ms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 10 ms = phasor-invalid guard ---
    if L < spc:
        logger.warning("L=%d < samples_per_cycle=%d -> phasor invalid; emitting '—'.", L, spc)
        run["result"] = "phasor_invalid"
        (out_dir / "cv_summary.json").write_text(json.dumps(run, indent=2))
        return

    # --- shared load (framework path) ---
    windows, labels, row_indices = load_windows_and_labels(config)
    if row_indices is not None:
        fpath = (
            Path(config.window_extraction.windows_local_dir)
            / f"X_fault_only_{config.dataset.topology}_W{str(config.window_extraction.window_length).replace('.', 'p')}.raw"
        )
        windows = write_filtered_memmap(windows, row_indices, fpath)
    feature_names = load_meta_data(config)["feature_names"]
    registry = LineRegistry(feature_names)
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, config)
    gkf = GroupKFold(n_splits=int(config.training.n_splits))
    logger.info(
        "Loaded windows=%s labels=%d task=%s W=%dms", windows.shape, len(labels), task, window_ms
    )

    if task == "fc":
        _run_fc(config, bcfg, windows, labels, y, sample_ids, gkf, registry, spc, run, out_dir)
    elif task in ("fl_two_ended", "fl_one_ended"):
        _run_fl(
            config, bcfg, windows, labels, y, sample_ids, gkf, registry, spc, run, out_dir, task
        )
    else:
        raise ValueError(f"Unknown baseline.task {task!r}")


def _run_fc(config, bcfg, windows, labels, y, sample_ids, gkf, registry, spc, run, out_dir):
    clf = ConventionalFaultClassifier(registry, spc, bcfg.fc.pickup_grid, bcfg.fc.ground_grid)
    logger.info("Precomputing phase-selector features over %d windows ...", len(labels))
    feat = clf.precompute(windows, labels)

    fold_f1, fold_rows = [], []
    for i, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups=sample_ids), start=1):
        feat_tr = {k: v[tr] for k, v in feat.items()}
        feat_te = {k: v[te] for k, v in feat.items()}
        best = clf.fit(feat_tr, y[tr])
        pred_te = clf.predict(feat_te)
        f1 = f1_score(y[te], pred_te, average="macro", zero_division=0)
        # fault-only macro-F1 (drop true no_fault rows) for the Fig.3 companion
        from psp_helper.constants import FAULT_LABEL_TO_ID

        nf = FAULT_LABEL_TO_ID["no_fault"]
        fmask = y[te] != nf
        f1_fault = (
            f1_score(y[te][fmask], pred_te[fmask], average="macro", zero_division=0)
            if fmask.any()
            else float("nan")
        )
        fold_f1.append(float(f1))
        fold_rows.append(
            {
                "fold": i,
                "f1_macro": float(f1),
                "f1_macro_fault_only": float(f1_fault),
                "tau_p": best["tau_p"],
                "tau_g": best["tau_g"],
                "train_f1": best["f1"],
            }
        )
        logger.info(
            "Fold %d: macro-F1=%.4f (fault-only %.4f) tau_p=%.2f tau_g=%.2f",
            i,
            f1,
            f1_fault,
            best["tau_p"],
            best["tau_g"],
        )

    run["metric"] = "macro_f1"
    run["f1_macro"] = _fold_stats(fold_f1)
    run["f1_macro_fault_only"] = _fold_stats([r["f1_macro_fault_only"] for r in fold_rows])
    _write(out_dir, run, fold_rows)
    logger.info(
        "FC phase selector: macro-F1 = %.4f ± %.4f",
        run["f1_macro"]["mean"],
        run["f1_macro"]["std"],
    )


def _run_fl(config, bcfg, windows, labels, y, sample_ids, gkf, registry, spc, run, out_dir, task):
    params_df = pd.read_csv(bcfg.labels_csv_path, index_col="sample_id")
    if task == "fl_two_ended":
        est = TwoEndedFaultLocator(
            registry,
            spc,
            current_sign=bcfg.fl.current_sign,
            min_current_ratio=bcfg.fl.min_current_ratio,
        )
    else:
        est = SingleEndedFaultLocator(
            registry, spc, neglect_mutual=bcfg.fl.neglect_mutual, terminal=bcfg.fl.terminal
        )
    logger.info("Predicting %s over %d onset windows ...", task, len(labels))
    pred = est.predict(windows, labels, params_df)  # % line length; nan where degenerate
    y_true = np.asarray(y, dtype=float)

    # Per-episode "settled" latched estimate: for each episode, use the onset window with
    # the largest faulted-phase current (what a distance/location element would actually
    # latch). This is the faithful conventional protocol; the per-window number below is
    # the "identical-protocol" comparison and is sensitive to un-settled onset windows.
    settle = faulted_phase_current_mag(windows, labels, registry, spc)
    ep = pd.DataFrame({"sid": sample_ids, "settle": settle, "pred": pred, "true": y_true})
    best = ep.loc[ep.groupby("sid")["settle"].idxmax()]

    fold_mae, fold_mae_settled, fold_rows = [], [], []
    for i, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups=sample_ids), start=1):
        mae, n_valid = _mae_ignore_nan(y_true[te], pred[te])
        cover = n_valid / max(len(te), 1)
        te_sids = set(np.unique(sample_ids[te]).tolist())
        bsub = best[best["sid"].isin(te_sids)]
        smae, _ = _mae_ignore_nan(bsub["true"].to_numpy(), bsub["pred"].to_numpy())
        fold_mae.append(mae)
        fold_mae_settled.append(smae)
        fold_rows.append(
            {
                "fold": i,
                "mae": mae,
                "mae_settled": smae,
                "coverage": float(cover),
                "n_test": int(len(te)),
                "n_episodes": int(len(bsub)),
            }
        )
        logger.info(
            "Fold %d: per-window MAE=%.3f%% | per-episode-settled MAE=%.3f%% (n=%d, eps=%d)",
            i,
            mae,
            smae,
            len(te),
            len(bsub),
        )

    run["metric"] = "mae_percent_line"
    run["mae"] = _fold_stats(fold_mae)
    run["mae_settled_per_episode"] = _fold_stats(fold_mae_settled)
    run["coverage"] = _fold_stats([r["coverage"] for r in fold_rows])
    # sanity diagnostics (help validate terminal/units/sign)
    ok = ~np.isnan(pred)
    run["diagnostics"] = {
        "overall_mae": (
            float(mean_absolute_error(y_true[ok], pred[ok])) if ok.any() else float("nan")
        ),
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
        "pred_mean": float(np.nanmean(pred)),
        "frac_valid": float(ok.mean()),
        "corr_pred_true": (
            float(np.corrcoef(pred[ok], y_true[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        ),
    }
    _write(out_dir, run, fold_rows)
    logger.info(
        "%s: per-window MAE = %.3f%% ± %.3f%% | per-episode-settled MAE = %.3f%% ± %.3f%% (corr %.3f)",
        task,
        run["mae"]["mean"],
        run["mae"]["std"],
        run["mae_settled_per_episode"]["mean"],
        run["mae_settled_per_episode"]["std"],
        run["diagnostics"]["corr_pred_true"],
    )


def _write(out_dir: Path, run: dict, fold_rows: List[dict]) -> None:
    (out_dir / "cv_summary.json").write_text(json.dumps(run, indent=2))
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_results.csv", index=False)
    logger.info("Wrote %s", out_dir / "cv_summary.json")


if __name__ == "__main__":
    main()
