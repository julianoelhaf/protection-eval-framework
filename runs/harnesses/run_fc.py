"""General FC regeneration harness (pinned-pipeline).

Reuses the exact model factory, per-fold StandardScaler, and deterministic
GroupKFold split of the framework. Accepts arbitrary Hydra overrides so the
same script covers timing horizons and the App-B hyperparameter ablation grid.

Emits a full per-run lifecycle log to stdout (captured in the SLURM .out run
log): RUN/ENV -> LOAD -> DATA -> per-fold FOLD (train/test sizes, scaler/fit/
predict timing, per-fold metric) -> the parsed metric line -> DONE. The metric
line alone is appended to RUN_OUT (kept clean for the dashboard parser).

Usage:
    python run_fc.py <tag> <hydra_override> [<hydra_override> ...]
Env:
    WINDIR     -> window_extraction.windows_local_dir (default: dataset dir)
    RUN_OUT  -> append log file (metric line only)
"""
import copy
import gc
import os
import sys
import time
from time import perf_counter

import numpy as np
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CFG_DIR = "/path/to/repos/protection-eval-framework/config"
WINDIR = os.environ.get("WINDIR", "/path/to/datasets/PROTECT-90/windows/PROTECT-90")
RESULTS = os.environ.get("RUN_OUT", "/path/to/scratch/run_fc_results.txt")


def p(tag, stage, **kv):
    """Per-run lifecycle line -> stdout only (captured in the committed SLURM .out)."""
    kvs = " ".join("%s=%s" % (k, v) for k, v in kv.items())
    print(("[%s] %s %s" % (tag, stage, kvs)).rstrip(), flush=True)


def log(line):
    """Final metric line -> stdout + the parsed RUN_OUT results file (format unchanged)."""
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def main():
    tag = sys.argv[1]
    extra = list(sys.argv[2:])
    t_start = time.time()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(
            config_name="main-config",
            overrides=[
                f"window_extraction.windows_local_dir={WINDIR}",
                "tracking.mode=disabled",
            ] + extra,
        )
    import sklearn
    p(tag, "RUN", overrides=extra)
    p(tag, "ENV", sklearn=sklearn.__version__, python=sys.version.split()[0],
      seed=int(cfg.training.random_state), n_splits=int(cfg.training.n_splits))

    from fcl_psp.models.model_utils import create_model_from_name, get_task_type
    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
    from psp_helper.windows_helper import load_windows_and_labels

    p(tag, "LOAD", windir=WINDIR)
    windows, labels, ri = load_windows_and_labels(cfg)
    if ri is not None:
        fp = os.path.join(WINDIR, f"X_fo_run_{tag}.raw")
        windows = write_filtered_memmap(windows, ri, fp)
        p(tag, "FILTER", fault_only_rows=len(ri))
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    N, L, F = windows.shape
    task = get_task_type(cfg)
    nclass = len(set(y.tolist())) if task in ("binary", "multiclass") else "NA"
    p(tag, "DATA", N=N, L=L, F=F, task=task, n_classes=nclass,
      target=cfg.training.target_label, model=cfg.model.model_name)

    model = create_model_from_name(cfg)
    gkf = GroupKFold(n_splits=int(cfg.training.n_splits))
    t0 = time.time()
    scores = []
    for i, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), 1):
        Xtr = np.asarray(windows[tr]).reshape(len(tr), L * F)
        sc = StandardScaler(copy=False)
        ts = perf_counter()
        Xtr = sc.fit_transform(Xtr)
        scaler_s = perf_counter() - ts
        tf = perf_counter()
        m = copy.deepcopy(model).fit(Xtr, y[tr])
        fit_s = perf_counter() - tf
        del Xtr
        gc.collect()
        Xte = sc.transform(np.asarray(windows[te]).reshape(len(te), L * F))
        tp = perf_counter()
        pred = m.predict(Xte)
        pred_s = perf_counter() - tp
        del Xte, m
        gc.collect()
        if task in ("binary", "multiclass"):
            s = float(f1_score(y[te], pred, average="macro", zero_division=0))
            name = "macroF1"
        else:
            s = float(mean_absolute_error(y[te], pred))
            name = "MAE"
        scores.append(s)
        p(tag, "FOLD", i=i, n_train=len(tr), n_test=len(te), scaler_s=round(scaler_s, 1),
          fit_s=round(fit_s, 1), predict_s=round(pred_s, 2), **{name: round(s, 4)})
    arr = np.array(scores)
    log(f"[{tag}] {name} mean={arr.mean():.4f} std={arr.std():.4f} "
        f"per_fold={[round(x, 4) for x in scores]} overrides={extra} ({time.time() - t0:.0f}s)")
    p(tag, "DONE", total_s=round(time.time() - t_start))


if __name__ == "__main__":
    main()
