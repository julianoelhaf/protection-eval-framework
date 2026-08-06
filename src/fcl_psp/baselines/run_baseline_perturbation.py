"""Conventional protection baselines under measurement-fidelity degradation.

Runs the vendored DSP baselines (two-ended / one-ended fault location, and the
symmetrical-component phase selector) through the SAME degradation axes (additive
Gaussian noise, CT saturation, sync jitter) and the SAME seeded per-(fold, axis,
level, realization) protocol as the ML fidelity evaluator (run_perturbation_eval.py),
so the baseline robustness curves are directly comparable to the learning models'.

Design (verified against run_conventional_baselines.py + run_perturbation_eval.py):
  * No training / no scaling -- the estimators work in physical volts/amps.
  * The DFT front end is recomputed on the PERTURBED windows every cell.
  * The FC phase-selector thresholds (tau_p, tau_g) are grid-fit on the CLEAN train
    fold and applied to the perturbed test fold (train-only discipline preserved).
  * 10 ms windows are phasor-invalid (L < samples_per_cycle) -> skipped with a stub.
  * GroupKFold by episode is identical to the reference runs, so the clean level
    reproduces the committed clean baseline (identity anchor).

Output mirrors run_perturbation_eval exactly:
  reports/perturbation/baseline_<task>_W<w>ms/{perturbation_summary.csv,
  perturbation_run_meta.json, perturbation_metrics.parquet}
with summary columns axis,level,metric,mean,std,count. FL emits metrics
`mae` (per-window) + `mae_settled` (per-episode); FC emits `macro_f1` +
`macro_f1_fault_only`.

Usage:
    python -m fcl_psp.baselines.run_baseline_perturbation \
        baseline.task=fl_two_ended training.target_label=y_fault_location \
        window_extraction.window_length=0.020
    (task=fc uses training.target_label=event_type)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from psp_helper.config import MainConfig
from psp_helper.windows_helper import load_meta_data, load_windows_and_labels
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from fcl_psp.baselines.data.line_registry import LineRegistry
from fcl_psp.baselines.estimators import (
    ConventionalFaultClassifier,
    SingleEndedFaultLocator,
    TwoEndedFaultLocator,
    faulted_phase_current_mag,
)
from fcl_psp.baselines.run_conventional_baselines import _load_baseline_cfg, _mae_ignore_nan
from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
from fcl_psp.models.run_perturbation_eval import _is_identity, _load_perturbation_cfg
from fcl_psp.perturbation.operators import apply_axis, channel_layout
from fcl_psp.perturbation.seeding import make_rng

logger = logging.getLogger("run_baseline_perturbation")


@hydra.main(version_base=None, config_path="../../../config", config_name="main-config.yaml")
def main(config: MainConfig) -> None:
    bcfg = _load_baseline_cfg(config)
    pcfg = _load_perturbation_cfg(config)
    task = bcfg.task
    fs = int(config.dataset.sampling_frequency)
    window_ms = round(config.window_extraction.window_length * 1000)
    spc = int(bcfg.samples_per_cycle)
    L = round(config.window_extraction.window_length * fs)
    out_dir = Path(pcfg.out_dir) / f"baseline_{task}_W{window_ms}ms"
    out_dir.mkdir(parents=True, exist_ok=True)

    if L < spc:  # phasor-invalid (e.g. 10 ms) -- no fundamental cycle
        (out_dir / "perturbation_summary.csv").write_text("axis,level,metric,mean,std,count\n")
        json.dump(
            {"result": "phasor_invalid", "task": task, "window_ms": window_ms},
            open(out_dir / "perturbation_run_meta.json", "w"),
            indent=2,
        )
        logger.warning("phasor-invalid (L=%d < spc=%d); wrote stub -> %s", L, spc, out_dir)
        print(f"BLP_SKIP phasor_invalid {out_dir}")
        return

    windows, labels, ri = load_windows_and_labels(config)
    if ri is not None:  # FL: filter to fault-present rows, aligned with labels
        fp = Path(config.window_extraction.windows_local_dir) / f"X_fo_blp_{task}_W{window_ms}.raw"
        windows = write_filtered_memmap(windows, ri, fp)
    feature_names = load_meta_data(config)["feature_names"]
    registry = LineRegistry(feature_names)
    layout = channel_layout(feature_names)
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, config)
    sample_ids = np.asarray(sample_ids)
    labels = labels.reset_index(drop=True)
    N, Lw, F = windows.shape
    gkf = GroupKFold(n_splits=int(config.training.n_splits))

    rows = []

    def sweep(fold, te, predict_cell):
        Wte3 = np.asarray(windows[te]).reshape(len(te), Lw, F)
        for axis in pcfg.axes:
            for level in pcfg.levels_for(axis):
                reps = 1 if _is_identity(axis, level) else int(pcfg.n_realizations)
                for rep in range(reps):
                    rng = make_rng(pcfg.seed_global, fold, axis, level, rep)
                    Wp3 = apply_axis(axis, level, Wte3.copy(), layout, rng)
                    for metric, value in predict_cell(Wp3).items():
                        rows.append(
                            {
                                "task": task,
                                "window_ms": window_ms,
                                "axis": axis,
                                "level": str(level),
                                "realization": rep,
                                "fold": fold,
                                "metric": metric,
                                "value": value,
                                "identity": bool(_is_identity(axis, level)),
                            }
                        )

    if task == "fc":
        from psp_helper.constants import FAULT_LABEL_TO_ID

        nf = FAULT_LABEL_TO_ID["no_fault"]
        clf = ConventionalFaultClassifier(registry, spc, bcfg.fc.pickup_grid, bcfg.fc.ground_grid)
        feat_all = clf.precompute(windows, labels)  # clean features, for fitting only
        for fold, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), 1):
            clf.fit({k: v[tr] for k, v in feat_all.items()}, y[tr])  # freeze tau on clean train
            lab_te = labels.iloc[te].reset_index(drop=True)
            yte = np.asarray(y[te])

            def predict_cell(Wp3, lab_te=lab_te, yte=yte):
                pred = clf.predict(clf.precompute(Wp3, lab_te))
                out = {"macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0))}
                keep = yte != nf
                if keep.any():
                    out["macro_f1_fault_only"] = float(
                        f1_score(yte[keep], pred[keep], average="macro", zero_division=0)
                    )
                return out

            sweep(fold, te, predict_cell)
    else:
        params_df = pd.read_csv(bcfg.labels_csv_path, index_col="sample_id")
        if task == "fl_two_ended":
            est = TwoEndedFaultLocator(
                registry,
                spc,
                current_sign=bcfg.fl.current_sign,
                min_current_ratio=bcfg.fl.min_current_ratio,
            )
        elif task == "fl_one_ended":
            est = SingleEndedFaultLocator(
                registry, spc, neglect_mutual=bcfg.fl.neglect_mutual, terminal=bcfg.fl.terminal
            )
        else:
            raise ValueError(f"unknown baseline.task={task}")
        for fold, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), 1):
            lab_te = labels.iloc[te].reset_index(drop=True)
            yte = np.asarray(y[te], dtype=float)
            sids_te = sample_ids[te]

            def predict_cell(Wp3, lab_te=lab_te, yte=yte, sids_te=sids_te):
                pred = est.predict(Wp3, lab_te, params_df)
                mae_w, _ = _mae_ignore_nan(yte, pred)
                settle = faulted_phase_current_mag(Wp3, lab_te, registry, spc)
                ep = pd.DataFrame({"sid": sids_te, "settle": settle, "pred": pred, "true": yte})
                best = ep.loc[ep.groupby("sid")["settle"].idxmax()]
                mae_s, _ = _mae_ignore_nan(best["true"].to_numpy(), best["pred"].to_numpy())
                return {"mae": mae_w, "mae_settled": mae_s}

            sweep(fold, te, predict_cell)

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "perturbation_metrics.parquet", index=False)
    summary = (
        df.groupby(["axis", "level", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["axis", "level"])
    )
    summary.to_csv(out_dir / "perturbation_summary.csv", index=False)
    json.dump(
        {
            "task": task,
            "baseline": task,
            "window_ms": window_ms,
            "seed_global": pcfg.seed_global,
            "n_realizations": pcfg.n_realizations,
            "axes": list(pcfg.axes),
            "n_splits": int(config.training.n_splits),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "conventional protection baselines under measurement-fidelity degradation",
        },
        open(out_dir / "perturbation_run_meta.json", "w"),
        indent=2,
    )
    logger.info("wrote %s (%d rows)", out_dir, len(df))
    print(f"BLP_DONE {out_dir} rows={len(df)}")


if __name__ == "__main__":
    main()
