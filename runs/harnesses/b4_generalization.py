"""B4: within-PROTECT-90 operating-point-shift generalization sweep.

Tests whether the learning models generalize under a distribution shift in a
physically meaningful operating-point variable (fault resistance R_f), rather
than only under in-distribution episode-grouped CV. The split is by EPISODE
(no window leakage): train on one R_f range, evaluate on the disjoint shifted
range. Same pipeline (model factory, train-only StandardScaler, onset-only
validity for FL) as the reference runs.

Emits a full per-run lifecycle log to stdout (the committed SLURM .out):
RUN/ENV -> LOAD -> SPLIT (R_f threshold, train/test sizes, leakage guard) ->
FIT -> EVAL -> the parsed metric line -> DONE. Only the metric line is appended
to B4_OUT.

Usage:
    python b4_generalization.py <tag> <target> <model_name> <wl> <direction>
      direction: high  -> train R_f <= 80th pct, test R_f > 80th pct
                 low   -> train R_f >= 20th pct, test R_f < 20th pct
"""
import copy
import gc
import os
import sys
import time
from time import perf_counter

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

CFG_DIR = "/path/to/repos/protection-eval-framework/config"
WINDIR = os.environ.get("WINDIR", "/path/to/datasets/PROTECT-90/windows/PROTECT-90")
CSV = os.environ.get("LABELS_CSV", "/path/to/datasets/PROTECT-90/hv_double_line_90kv_labels.csv")
RESULTS = os.environ.get("B4_OUT", "/path/to/scratch/b4_results.txt")


def p(tag, stage, **kv):
    """Per-run lifecycle line -> stdout only (captured in the committed SLURM .out)."""
    kvs = " ".join("%s=%s" % (k, v) for k, v in kv.items())
    print(("[%s] %s %s" % (tag, stage, kvs)).rstrip(), flush=True)


def log(line):
    """Final metric line -> stdout + the parsed B4_OUT results file (format unchanged)."""
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def bootstrap_ci(sample_ids_te, y_te, pred, task, B=1000, seed=42):
    """Episode-level bootstrap 95% CI on the shifted-test metric.

    The R_f split is a deterministic threshold, so dispersion comes from test
    sampling: resample the disjoint test EPISODES with replacement (the unit of
    independence), recompute the metric over their windows. Uniform across
    deterministic and stochastic models; leaves the point estimate untouched.
    """
    eps = np.asarray(sample_ids_te)
    uniq = np.unique(eps)
    ep2pos = {}
    for i, e in enumerate(eps):
        ep2pos.setdefault(e, []).append(i)
    ep2pos = {e: np.asarray(v) for e, v in ep2pos.items()}
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([ep2pos[e] for e in samp])
        if task in ("binary", "multiclass"):
            vals[b] = f1_score(y_te[idx], pred[idx], average="macro", zero_division=0)
        else:
            vals[b] = mean_absolute_error(y_te[idx], pred[idx])
    return (float(np.mean(vals)), float(np.std(vals)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    tag, target, model_name, wl, direction = sys.argv[1:6]
    seed = sys.argv[6] if len(sys.argv) > 6 else None
    t_start = time.time()
    overrides = [
        f"training.target_label={target}",
        f"model.model_name={model_name}",
        f"window_extraction.window_length={wl}",
        f"window_extraction.windows_local_dir={WINDIR}",
        "tracking.mode=disabled",
    ]
    if seed is not None:  # override the model random_state to sample the training distribution
        overrides.append(f"training.random_state={seed}")
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="main-config", overrides=overrides)
    import sklearn
    p(tag, "RUN", target=target, model=model_name, wl=wl, direction=direction)
    p(tag, "ENV", sklearn=sklearn.__version__, python=sys.version.split()[0], seed=int(cfg.training.random_state))

    from fcl_psp.models.model_utils import create_model_from_name, get_task_type
    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
    from psp_helper.windows_helper import load_windows_and_labels

    p(tag, "LOAD", windir=WINDIR)
    windows, labels, ri = load_windows_and_labels(cfg)
    if ri is not None:
        windows = write_filtered_memmap(windows, ri, os.path.join(WINDIR, f"X_fo_b4_{tag}.raw"))
        p(tag, "FILTER", fault_only_rows=len(ri))
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    sample_ids = np.asarray(sample_ids)

    rf = pd.read_csv(CSV, index_col="sample_id")["fault_resistance"]
    uniq = np.unique(sample_ids)
    rf_ep = rf.reindex(uniq).values
    rf_win = rf.reindex(sample_ids).values

    if direction == "high":
        thr = np.nanpercentile(rf_ep, 80)
        tr_mask, te_mask = rf_win <= thr, rf_win > thr
    elif direction == "low":
        thr = np.nanpercentile(rf_ep, 20)
        tr_mask, te_mask = rf_win >= thr, rf_win < thr
    else:
        raise ValueError(direction)

    tr, te = np.where(tr_mask)[0], np.where(te_mask)[0]
    leak = bool(set(sample_ids[tr]) & set(sample_ids[te]))
    p(tag, "SPLIT", shift=direction, thr=round(float(thr), 3), n_train=len(tr), n_test=len(te),
      train_episodes=len(set(sample_ids[tr])), test_episodes=len(set(sample_ids[te])), episode_leakage=leak)
    assert not leak, "episode leakage across R_f split"

    N, L, F = windows.shape
    task = get_task_type(cfg)
    p(tag, "DATA", N=N, L=L, F=F, task=task)
    model = create_model_from_name(cfg)
    t0 = time.time()
    Xtr = np.asarray(windows[tr]).reshape(len(tr), L * F)
    sc = StandardScaler(copy=False)
    Xtr = sc.fit_transform(Xtr)
    tf = perf_counter()
    m = copy.deepcopy(model).fit(Xtr, y[tr])
    p(tag, "FIT", fit_s=round(perf_counter() - tf, 1))
    del Xtr
    gc.collect()
    Xte = sc.transform(np.asarray(windows[te]).reshape(len(te), L * F))
    pred = m.predict(Xte)
    if task in ("binary", "multiclass"):
        val, name = float(f1_score(y[te], pred, average="macro", zero_division=0)), "macroF1"
    else:
        val, name = float(mean_absolute_error(y[te], pred)), "MAE"
    p(tag, "EVAL", **{name: round(val, 4)})
    ci_str = ""
    if os.environ.get("BOOT", "0") == "1":  # optional test-set bootstrap (secondary to seed spread)
        bmean, bstd, blo, bhi = bootstrap_ci(sample_ids[te], y[te], pred, task)
        p(tag, "EVAL_CI", boot_mean=round(bmean, 4), boot_std=round(bstd, 4),
          ci_lo=round(blo, 4), ci_hi=round(bhi, 4))
        ci_str = f" boot_std={bstd:.4f} ci95=[{blo:.4f},{bhi:.4f}]"
    seed_str = f" seed={seed}" if seed is not None else ""
    log(f"[{tag}] {target} {model_name} W={wl} shift={direction}{seed_str} thr={thr:.3f} "
        f"n_train={len(tr)} n_test={len(te)} {name}={val:.4f}{ci_str} ({time.time()-t0:.0f}s)")
    p(tag, "DONE", total_s=round(time.time() - t_start))


if __name__ == "__main__":
    main()
