"""Runtime regeneration harness (pinned-pipeline, single node).

Mirrors run_fc.py's data handling + the run_model_runtime.py per-fold
protocol, REUSING the repo's own measure_predict_runtime() so the predict-side
methodology is identical. Full observability (no relay reduction).

Reports (mean over the 5 GroupKFold folds):
    train_s   = model.fit time            -> table "Train [s]"
    infer_us  = predict batch per-sample  -> table "Infer. [us]"  (x1e6)
    thr_ks    = predict throughput/1000   -> table "Thr. [k/s]"
Also writes per-fold detail to runtime_csv/<tag>.csv and records the node.

Emits a full per-run lifecycle log to stdout (the committed SLURM .out):
RUN/ENV/NODE -> LOAD -> DATA -> per-fold FOLD (fit + throughput) -> the parsed
metric line -> DONE. Only the metric line is appended to RUN_OUT.

Usage:
    python run_runtime.py <tag> <hydra_override> ...
Env: WINDIR, RUN_OUT
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CFG_DIR = "/path/to/repos/protection-eval-framework/config"
WINDIR = os.environ.get("WINDIR", "/path/to/datasets/PROTECT-90/windows/PROTECT-90")
RESULTS = os.environ.get("RUN_OUT", "/path/to/scratch/run_runtime_results.txt")
CSVDIR = "/path/to/scratch/runtime_csv"


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
    p(tag, "NODE", node=os.uname().nodename, cpus=os.cpu_count(),
      omp=os.environ.get("OMP_NUM_THREADS", "?"))

    from fcl_psp.models.model_utils import create_model_from_name, get_task_type
    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
    from fcl_psp.models.run_model_runtime import measure_predict_runtime
    from psp_helper.windows_helper import load_windows_and_labels

    p(tag, "LOAD", windir=WINDIR)
    windows, labels, ri = load_windows_and_labels(cfg)
    if ri is not None:  # FL: materialize fault-only rows (full 48 features)
        fp = os.path.join(WINDIR, f"X_fo_rt_{tag}.raw")
        windows = write_filtered_memmap(windows, ri, fp)
        p(tag, "FILTER", fault_only_rows=len(ri))
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    N, L, F = windows.shape
    task = get_task_type(cfg)
    p(tag, "DATA", N=N, L=L, F=F, task=task, target=cfg.training.target_label, model=cfg.model.model_name)
    model = create_model_from_name(cfg)
    gkf = GroupKFold(n_splits=int(cfg.training.n_splits))

    rows = []
    t_all = time.time()
    for i, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), start=1):
        Xtr = np.asarray(windows[tr]).reshape(len(tr), L * F)
        sc = StandardScaler(copy=False)
        Xtr = sc.fit_transform(Xtr)
        m = copy.deepcopy(model)
        t0 = perf_counter()
        m.fit(Xtr, y[tr])
        fit_s = perf_counter() - t0
        del Xtr
        gc.collect()
        Xte = np.asarray(windows[te]).reshape(len(te), L * F)
        t0 = perf_counter()
        Xte = sc.transform(Xte)
        sca_test_s = perf_counter() - t0
        pr = measure_predict_runtime(m, Xte, n_repeats=30, n_warmup=5, max_batch_samples=2048)
        rows.append({"fold": i, "n_test": int(len(te)), "fit_s": fit_s,
                     "sca_test_s": sca_test_s, **pr})
        p(tag, "FOLD", i=i, n_train=len(tr), n_test=len(te), fit_s=round(fit_s, 1),
          infer_us=round(pr["predict_batch_per_sample_mean_s"] * 1e6, 2),
          thr_ks=round(pr["predict_throughput_samples_per_s"] / 1000.0, 1))
        del Xte, m
        gc.collect()

    rdf = pd.DataFrame(rows)
    train_s = float(rdf["fit_s"].mean())
    infer_us = float(rdf["predict_batch_per_sample_mean_s"].mean() * 1e6)
    thr_ks = float(rdf["predict_throughput_samples_per_s"].mean() / 1000.0)
    os.makedirs(CSVDIR, exist_ok=True)
    rdf.to_csv(os.path.join(CSVDIR, f"{tag}.csv"), index=False)
    log(
        f"[{tag}] train_s={train_s:.1f} infer_us={infer_us:.2f} thr_ks={thr_ks:.1f} "
        f"nfeat={F} node={os.uname().nodename} cpus={os.cpu_count()} "
        f"omp={os.environ.get('OMP_NUM_THREADS','?')} overrides={extra} ({time.time() - t_all:.0f}s)"
    )
    p(tag, "DONE", total_s=round(time.time() - t_start))


if __name__ == "__main__":
    main()
