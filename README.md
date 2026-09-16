# A Standardized Framework for Machine Learning in Power System Protection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0%20(pinned)-orange.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ijepes.2026.112169-blue.svg)](https://doi.org/10.1016/j.ijepes.2026.112169)

Companion code and reproducibility harness for the paper **"A Standardized Framework for Machine
Learning in Power System Protection"**, published in the *International Journal of Electrical Power &
Energy Systems*, vol. 181, art. 112169 (2026), DOI: [10.1016/j.ijepes.2026.112169](https://doi.org/10.1016/j.ijepes.2026.112169).

The framework trains and evaluates models for **fault classification (FC)** and **fault localization (FL)** on
windowed voltage/current measurements from the [PROTECT-90 benchmark](https://doi.org/10.5281/zenodo.21109169),
under one standardized protocol — identical sensing, decision horizons of 10–50 ms, and episode-grouped
cross-validation. Every reported number is regenerated under a pinned environment and traced end-to-end to the
committed evidence under `reports/`.

> The importable Python package is **`fcl_psp`**; the repository directory is `protection-eval-framework`.

---

## Overview

- **Benchmark — PROTECT-90.** 9022 electromagnetic-transient (EMT) episodes, one 90 kV double-line topology,
  f_s = 6400 Hz, 8 protection relays × 6 channels = 48 features. Archived on Zenodo
  ([DOI 10.5281/zenodo.21109169](https://doi.org/10.5281/zenodo.21109169)).
- **Tasks.** FC — 11-class `event_type`, reported as macro-F1 (severe class imbalance → macro, not accuracy);
  FL — `y_fault_location` (normalized line position), reported as MAE in % of line length.
- **Protocol (frozen).** Deterministic `GroupKFold(n_splits=5)` grouped by simulation episode (no shuffle);
  per-fold **train-only** `StandardScaler`; fixed default hyper-parameters (no tuning); FL keeps only
  fault-onset windows. `scikit-learn` is hard-pinned to **1.8.0**.
- **Reproducibility.** The `runs/` harnesses regenerate every paper number under the pinned environment and
  commit one-line metric files under `reports/`, so every reported value can be traced back to the run that
  produced it.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/fcl_psp/` | The Python package: `models/` (core pipeline, runtime, perturbation eval, posthoc), `baselines/` (conventional protection baselines), `perturbation/` (measurement-fidelity operators), `evaluation/` (exports, diagnostics). |
| `config/` | Hydra config tree (`main-config.yaml` + groups: `dataset`, `model`, `training`, `window_extraction`, `tracking`, `ablation`, `baseline`, `perturbation`, `data_sparsity`). |
| `runs/` | Regeneration harnesses (`harnesses/`) and the SLURM launchers + job lists that invoked them (`drivers/`). See [`runs/README.md`](runs/README.md). |
| `reports/` | Committed provenance artifacts backing every reported number: `runs/*.txt`, `baselines/*/cv_summary.json`, `perturbation/*/perturbation_summary.csv`, `moment/`. |
| `tests/` | Dataset-free unit tests (label construction, metrics, model factory, baseline/perturbation operators). |
| `notebooks/` | Analysis and figure notebooks; exported vector assets in `notebooks/figures/`. |
| `hpc/` | SLURM scripts for the earlier full-campaign runs. |
| `paper/tables/` | Table generators: each numeric table in the manuscript is rebuilt from the committed evidence in `reports/` into a drop-in `table__<name>.tex`. See [`paper/tables/README.md`](paper/tables/README.md). |

---

## Installation

The reference environment is Python **3.12.4** with `scikit-learn==1.8.0` (hard-pinned — the reported FC/FL
numbers depend on this exact version; do not float it). The framework depends on the companion package
[`psp_helper`](https://github.com/julianoelhaf/power-grid-and-ai-helper-functions), which provides the config
schema, label constants, the Eq. 5 onset-validity rule, and the windowed-data loaders. It is not on PyPI, so it
is installed directly from its repository, pinned to an immutable commit.

```bash
# 1) companion package, pinned to the immutable commit the released pipeline runs against
pip install "psp_helper @ git+https://github.com/julianoelhaf/power-grid-and-ai-helper-functions.git@2c78db4"

# 2) this package (+ dev tools for tests and linting)
pip install -e ".[dev]"      # or: make requirements
```

`pyproject.toml` declares `psp_helper` as a pinned git dependency, so step 2 alone pulls it in. For a
byte-for-byte environment, install from the full transitive pin:

```bash
pip install -r requirements-lock.txt   # exact env: Python 3.12.4, scikit-learn 1.8.0, numpy 2.5.1, ...
```

**Optional — 1D-CNN baseline (PyTorch).** The learned CNN baseline runs in a separate conda environment with
`torch` (kept out of the pinned scikit-learn env). It is launched with
`PYTHONPATH=src:runs/harnesses/_shims`, where `_shims/` holds no-op `wandb`/`tabulate` stubs so the framework
imports succeed without those packages. Only needed to reproduce the CNN rows.

---

## Data

Download [PROTECT-90](https://doi.org/10.5281/zenodo.21109169) and point the config at your local copy — either
on the command line (`dataset.dataset_directory=/path/to/protect90`) or by editing
`config/dataset/hv_double_line_90kv.yaml`.

Models consume **windowed** tensors, cached on disk as `X_<topology>_W<window>_S<stride>.raw` (float32 memmap)
plus `y_<topology>_W<window>_S<stride>.parquet` (e.g. `X_hv_double_line_90kv_W0p020_S0p005.raw`). PROTECT-90
ships already preprocessed (per-episode pickles), so only the windowing stage runs; it is produced by the
`psp_helper` companion (`create_windows`, parameterized by `WINDOW_LENGTH` / `STEP_LENGTH`). Window caches
are regenerated artifacts and are **not** tracked in git.

---

## Quick start

```bash
# Fault classification (11-class event type, macro-F1)
python src/fcl_psp/models/run_model.py \
    training.target_label=event_type \
    model.model_name=mlp_classifier \
    window_extraction.window_length=0.020

# Fault localization (normalized line position, MAE %line)
python src/fcl_psp/models/run_model.py \
    training.target_label=y_fault_location \
    model.model_name=mlp_regressor \
    window_extraction.window_length=0.050
```

`run_model.py` loads the windows, runs the frozen 5-fold episode-grouped CV with per-fold train-only
standardization, computes macro-F1 (FC) / MAE (FL), and writes out-of-fold predictions + posthoc analyses.
Experiment tracking (Weights & Biases) is optional — set `tracking.mode=disabled` to run fully offline.

---

## Experiments

The paper's experiment families and their entry points:

| Family | Entry point | Notes |
|--------|-------------|-------|
| Core benchmark (FC / FL, timing horizons) | `src/fcl_psp/models/run_model.py` | The standardized pipeline; overrides select task/model/window. |
| Conventional protection baselines | `src/fcl_psp/baselines/run_conventional_baselines.py` | `+baseline.task={fc\|fl_two_ended\|fl_one_ended}`; raw volts/amps, no scaler. |
| Measurement-fidelity degradation | `src/fcl_psp/models/run_perturbation_eval.py` | Clean per-fold models trained once, evaluated on perturbed test folds (noise / CT saturation / jitter); baseline analogue in `run_baseline_perturbation.py`. |
| Runtime profiling | `src/fcl_psp/models/run_model_runtime.py` | Train/inference timing (machine-relative). |
| Reduced observability, generalization, 1D-CNN, MOMENT-1-large | `runs/harnesses/*` | See below. |

---

## Reproducing the paper numbers

`runs/` holds the exact scripts that regenerate the reported numbers under the pinned environment, plus the
SLURM drivers that launched them and the committed console logs. Each harness appends a tagged one-line metric
to a file under `reports/runs/`; the dashboard reads those files.

| Harness (`runs/harnesses/`) | Regenerates | Output (`reports/runs/`) |
|---|---|---|
| `smoke_reproduce.py` | Reference FC/FL benchmark cells (20/50 ms) | `repro_fc.txt`, `repro_fl.txt` |
| `run_fc.py` | Timing sweep (10/30/40 ms) + hyperparameter ablations (FC & FL) | `run_fc_results.txt`, `run_tim10_results.txt` |
| `run_obs.py` | Reduced-observability sweep (single-relay / same-line pairs) | `run_obs_results.txt` |
| `run_runtime.py` | Runtime table | `run_runtime_results.txt` |
| `b4_generalization.py` | Operating-point-shift (fault-resistance) generalization | `b4_generalization*_results.txt` |
| `cnn_baseline.py` | 1D-CNN learned baseline (PyTorch) | `cnn_results.txt` |
| `moment_baseline.py` + `aggregate_moment_results.py` | MOMENT-1-large foundation-model baseline (frozen encoder → linear probe + MLP head); **one SLURM job per outer fold** with per-fold train-only preprocessing (no shared writable cache) | `moment_results.txt` (+ per-fold JSONs & out-of-fold predictions under `reports/moment/`) |
| `class_and_split_stats.py` | Class balance + per-fold test-set sizes | `class_split_stats.txt` |

Drivers under `runs/drivers/` follow a SLURM array + job-list pattern (`sed -n "${SLURM_ARRAY_TASK_ID}p"
<joblist>` → `TAG|<hydra overrides>`). They were recorded as-run and contain hardcoded cluster paths — adapt
`WINDIR` / `RUN_OUT` / `CFG_DIR` for another environment (see [`runs/README.md`](runs/README.md)).

---

## Provenance

Every reported number is backed by a committed artifact in this repository: the tagged one-line metric files
under `reports/runs/`, the cross-validation summaries under `reports/baselines/`, the perturbation sweeps under
`reports/perturbation/`, and the per-fold MOMENT results under `reports/moment/`. The generators in
`paper/tables/` read those files directly, so each manuscript table can be rebuilt from — and diffed against —
the evidence it came from.

The authors additionally maintain an internal results-audit dashboard that renders this evidence as a static
site and enforces link-level traceability. That generator is maintainer tooling and is **not** part of the
public release; the artifacts it consumes are the `reports/` files published here.

---

## Configuration

All configuration is [Hydra](https://hydra.cc); the root is
[`config/main-config.yaml`](config/main-config.yaml).

| Group | Key knobs |
|-------|-----------|
| `dataset` | `dataset_directory` (your PROTECT-90 path), `topology`, `sampling_frequency` (6400). |
| `training` | `target_label`, `n_splits` (5), `random_state` (42), `test_size`. |
| `model` | `model_name` + per-family hyper-parameters (`mlp.*`, `hgb.*`, …). |
| `window_extraction` | `window_length`, `step_length_seconds`, `fault_start_only`. |
| `tracking` | `mode` (`online` / `offline` / `disabled`), W&B `project` / `entity`. |
| `ablation` | relay observability (`mode`: `full` / `single_relay` / `relay_subset` / `drop_one_relay`). |
| `baseline`, `perturbation` | Experiment A / B configs (added via `+baseline.*` / `perturbation.*`). |
| `data_sparsity` | sensor / phase / relay failure masks (off by default). |

**Target labels:** `event_type` (FC, macro-F1), `y_fault_present` (binary detection), `y_fault_location`
(FL, MAE), `y_fault_line` (line ID). The full model registry (classifier/regressor families) is in
[`src/fcl_psp/models/model_utils.py`](src/fcl_psp/models/model_utils.py).

---

## Tests & CI

```bash
pytest            # dataset-free unit tests (or: make test)
make lint         # flake8 + isort + black (line length 99)
```

The authors' CI runs these same lint + test steps on every push; its configuration is specific to an internal
runner and is not part of the public release. The two commands above are the whole check.

---

## What is not in this public release

This repository is a scrubbed, history-free export of the authors' internal working repository. The following
are intentionally **not** included, and nothing published here depends on them:

- the manuscript LaTeX sources (to follow on publication) — the `paper/tables/` generators **are** included;
- the internal results-audit dashboard and its generator scripts;
- the raw SLURM per-job console logs (`runs/logs/`) — the aggregated results they produced are in `reports/`;
- internal planning and working notes;
- the CI configuration and the export tooling that produced this copy.

### Placeholders you must adapt

Site-specific absolute paths were replaced with placeholders throughout — chiefly
`/path/to/datasets`, `/path/to/scratch`, `/path/to/repos/` and `/path/to/conda`, plus
`/path/to/hf_cache` in the MOMENT drivers. They appear in `config/`, in the job scripts under
`hpc/` and `runs/drivers/`, and in the copy-pasteable usage examples in the `runs/harnesses/`
and `src/fcl_psp/` module docstrings; point them at your own data root, scratch space, checkout
and environment before running anything.

The companion package [`psp_helper`](https://github.com/julianoelhaf/power-grid-and-ai-helper-functions)
(config schema, constants, windowing incl. the onset-validity rule) lives in a separate
repository. `pyproject.toml` pins it to an immutable commit — adjust that requirement to
wherever you obtain it.

---

## Citation

If you use this code, please cite the framework paper:

```bibtex
@article{oelhaf2026standardized,
  author    = {Oelhaf, Julian and Kordowich, Georg and P{\'e}rez-Toro, Paula Andrea and
               Bergler, Christian and J{\"a}ger, Johann and Maier, Andreas and Bayer, Siming},
  title     = {A standardized framework for machine learning in power system protection},
  journal   = {International Journal of Electrical Power \& Energy Systems},
  volume    = {181},
  pages     = {112169},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.ijepes.2026.112169},
  url       = {https://doi.org/10.1016/j.ijepes.2026.112169}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

And please also cite the **PROTECT-90 dataset**:

```bibtex
@misc{kordowich2026protect90,
  author    = {Kordowich, Georg and Oelhaf, Julian and Bergler, Christian and
               Maier, Andreas and Bayer, Siming and J{\"a}ger, Johann},
  title     = {{PROTECT-90}: A Fault Dataset for Power System Protection},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.21109169},
  url       = {https://doi.org/10.5281/zenodo.21109169},
  note      = {Dataset. Concept DOI (all versions): 10.5281/zenodo.18418329}
}
```

Funding: Deutsche Forschungsgemeinschaft (DFG) — 535389056.

---

## Contact

Julian Oelhaf — [julian.oelhaf@fau.de](mailto:julian.oelhaf@fau.de) ·
[Website](https://lme.tf.fau.de/persons/julian-oelhaf/) ·
[GitHub](https://github.com/julianoelhaf)

## License

MIT. See [LICENSE](LICENSE).
