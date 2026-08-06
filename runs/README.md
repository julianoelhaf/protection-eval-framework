# Reproduction harnesses

These are the exact scripts that (re)generate the paper's reported numbers under the **pinned environment**
(`scikit-learn==1.8.0`, `requirements-lock.txt`), the deterministic episode-grouped `GroupKFold` split, and the
per-fold train-only `StandardScaler`. They are committed as **provenance**: the results-audit dashboard
(`docs/`) links every regenerated number to the metric-computing line in these files and to the output file the
number was read from.

> These scripts ran on the FAU HPC cluster and contain **hardcoded absolute paths** (`/path/to/scratch/...`,
> `/path/to/repos/protection-eval-framework/config`). They are recorded as-run, not as a portable
> CLI — adapt `CFG_DIR` / `WINDIR` / `RUN_OUT` (below) for another environment.

## Layout
- `harnesses/` — the Python that loads windows, runs the pinned CV pipeline, computes the metric, and appends a
  tagged line to a results file under `reports/runs/`.
- `drivers/` — the SLURM `sbatch` launchers + `*_cmds.txt` / `*_joblist.txt` job lists that invoked them.

## Environment variables (harnesses)
- `WINDIR` — window directory (default `/path/to/datasets/PROTECT-90/windows/PROTECT-90`).
- `RUN_OUT` — results file to append to (see map below).
- `CFG_DIR` — hardcoded to the repo `config/` (Hydra config root) inside each harness.

## Tag → paper table → output file map
| Harness | Driver | Output (`reports/runs/`) | Tags | Paper table(s) |
|---|---|---|---|---|
| `smoke_reproduce.py` | `slurm_fc_repro.sbatch`, `slurm_fl.sbatch` | `repro_fc.txt`, `repro_fl.txt` | `=== <target> <model> W=<w>` | `tab:reference_20ms`; timing 20/50 ms |
| `run_fc.py` | `run_fc.sbatch` + `run_joblist.txt` | `run_fc_results.txt` | `tim_{mlp,gb}_W{30,40}`, `abl_*` | `tab:fc_timing_sensitivity` (30/40 ms); App-B ablations |
| `run_fc.py` | `run_tim10.sbatch` + `tim10_cmds.txt` | `run_tim10_results.txt` | `tim10_{fc,fl}_{mlp,gb,knn,ridge}` | timing 10 ms row (FC+FL) |
| `run_obs.py` | `run_obs.sbatch` + `obs_cmds.txt` | `run_obs_results.txt` | `obs_{task}_{model}_{s#\|p#-#}_W{w}` | `tab:fc_observability`, `tab:fl_observability` |
| `run_runtime.py` | `run_runtime.sbatch` | `run_runtime_results.txt` | `rt_{fc,fl}_{mlp,gb}_W{20,50}` | `tab:fc_runtime_compact`, `tab:fl_runtime_compact` |
| `b4_generalization.py` | `b4.sbatch` + `b4_joblist.txt` | `b4_generalization_results.txt` | `b4_{fc,fl}_{mlp,gb}_{hi,lo}` | `tab:generalization_shift` |
| `build_windows_10ms.py` | `run_build10.sbatch` | (builds 10 ms windows) | — | prerequisite for `tim10_*` |
| `plot_perturbation.py` | — | (fidelity figures) | — | `tab:fidelity_*` figures |

The reduced-observability rows are **derived**: the paper's "single relay" cell is the mean of the 8
single-relay `obs_*_s0..s7` runs, and "relay pair" the mean over the same-line terminal pairs
(`[(0,2),(4,6),(1,3),(5,7)]`). The dashboard recomputes these aggregates and asserts they match the printed
value.

## Run logs (`logs/`)
The raw SLURM per-job console logs are committed under `runs/logs/<campaign>/` — the audit chain is
only complete if the run itself is inspectable, not just its parsed result. Each log records the run's tag,
the full Hydra override string, timestamps, the execution node, and any traceback.

| `logs/` subdir | Campaign | Driver | → results file |
|---|---|---|---|
| `run_fc/` | timing 30/40 ms + App-B ablations | `drivers/run_fc.sbatch` | `reports/runs/run_fc_results.txt` |
| `b4/` | operating-point-shift generalization | `drivers/b4.sbatch` | `reports/runs/b4_generalization_results.txt` |
| `obs/` | reduced-observability sweep | `drivers/run_obs.sbatch` | `reports/runs/run_obs_results.txt` |
| `runtime/`, `tim10/` | runtime + 10 ms timing | `run_runtime.sbatch` / `run_tim10.sbatch` | added on completion |

**Policy:** raw per-run logs are committed as each run finishes; parsed/aggregated result files are committed
once their array completes (so no partial aggregate is ever shown). The tag on line _N_ of a driver's
`*_cmds.txt` maps to array task _N_ → `logs/<campaign>/<name>_<N>.out`.

**Lifecycle logging.** The harnesses log the full run to stdout (→ the committed `.out`):
`RUN`/`ENV` (overrides, sklearn/python version, seed, folds) → `LOAD` → `DATA` (shapes, N, features, classes) →
per-fold `FOLD` (train/test sizes, scaler/fit/predict seconds, per-fold metric) → the parsed metric line →
`DONE`. The metric line alone is appended to the results file, so the dashboard parser is unaffected.
_Note:_ the **initial** campaign (`run_fc`, `b4`, `obs`) ran on the earlier lean harnesses, so its committed
logs show data-loading + the aggregate only; the lifecycle logging above applies to all subsequent runs
(the Phase-2 stride / FL-50 ms-ablation / diagnostics harnesses and any re-runs).
