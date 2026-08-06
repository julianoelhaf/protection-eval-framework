"""Shared (cluster-visible) memory-lean reproduction of reference ML metrics.

Reuses the exact model factory, per-fold StandardScaler, and deterministic
GroupKFold split. Writes per-fold + summary lines to $REPRO_OUT (append).
Window dir from $WINDIR. Run one (target, model, wl) per invocation.
"""
import copy
import gc
import os
import sys
import time

import numpy as np
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CFG_DIR = "/path/to/repos/protection-eval-framework/config"
WINDIR = os.environ.get("WINDIR", "/path/to/scratch/win_fl")
RESULTS = os.environ.get("REPRO_OUT", "/path/to/scratch/repro_results.txt")


def log(line):
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def run(target, model_name, wl):
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(
            config_name="main-config",
            overrides=[
                f"training.target_label={target}",
                f"model.model_name={model_name}",
                f"window_extraction.window_length={wl}",
                f"window_extraction.windows_local_dir={WINDIR}",
                "tracking.mode=disabled",
            ],
        )
    from fcl_psp.models.model_utils import create_model_from_name, get_task_type
    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
    from psp_helper.windows_helper import load_windows_and_labels

    windows, labels, ri = load_windows_and_labels(cfg)
    if ri is not None:
        fp = os.path.join(WINDIR, f"X_fault_only_repro_{target}_{str(wl).replace('.', 'p')}.raw")
        windows = write_filtered_memmap(windows, ri, fp)
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    N, L, F = windows.shape
    task = get_task_type(cfg)
    model = create_model_from_name(cfg)
    gkf = GroupKFold(n_splits=int(cfg.training.n_splits))
    t_all = time.time()
    scores = []
    for i, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), start=1):
        Xtr = np.asarray(windows[tr]).reshape(len(tr), L * F)
        sc = StandardScaler(copy=False)
        Xtr = sc.fit_transform(Xtr)
        m = copy.deepcopy(model).fit(Xtr, y[tr])
        del Xtr
        gc.collect()
        Xte = sc.transform(np.asarray(windows[te]).reshape(len(te), L * F))
        pred = m.predict(Xte)
        del Xte, m
        gc.collect()
        if task in ("binary", "multiclass"):
            s, name = float(f1_score(y[te], pred, average="macro", zero_division=0)), "macroF1"
        else:
            s, name = float(mean_absolute_error(y[te], pred)), "MAE"
        scores.append(s)
    arr = np.array(scores)
    log(f"=== {target} {model_name} W={wl}: {name} mean={arr.mean():.4f} std={arr.std():.4f} "
        f"per_fold={[round(x, 4) for x in scores]} ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
