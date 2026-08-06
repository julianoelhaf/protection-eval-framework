# A Standardized Framework for Machine Learning in Power System Protection

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0%20(pinned)-orange.svg)
![Paper](https://img.shields.io/badge/paper-under%20review%20%40%20IJEPES-informational.svg)

Companion code and reproducibility harness for the manuscript **"A Standardized Framework for Machine
Learning in Power System Protection"** (Elsevier *International Journal of Electrical Power & Energy Systems*,
manuscript **IJEPES-D-26-01815**, under review).

The framework trains and evaluates models for **fault classification (FC)** and **fault localization (FL)** on
windowed voltage/current measurements from the [PROTECT-90 benchmark](https://doi.org/10.5281/zenodo.18418330),
under one standardized protocol — identical sensing, decision horizons of 10–50 ms, and episode-grouped
cross-validation. Every reported number is regenerated under a pinned environment and traced end-to-end by a
results-audit dashboard.

> The importable Python package is **`fcl_psp`**; the repository directory is `protection-eval-framework`.

---

## Overview

- **Benchmark — PROTECT-90.** 9022 electromagnetic-transient (EMT) episodes, one 90 kV double-line topology,
  f_s = 6400 Hz, 8 protection relays × 6 channels = 48 features. Archived on Zenodo
  ([DOI 10.5281/zenodo.18418330](https://doi.org/10.5281/zenodo.18418330)).
- **Tasks.** FC — 11-class `event_type`, reported as macro-F1 (severe class imbalance → macro, not accuracy);
  FL — `y_fault_location` (normalized line position), reported as MAE in % of line length.
- **Protocol (frozen).** Deterministic `GroupKFold(n_splits=5)` grouped by simulation episode (no shuffle);
  per-fold **train-only** `StandardScaler`; fixed default hyper-parameters (no tuning); FL keeps only
  fault-onset windows. `scikit-learn` is hard-pinned to **1.8.0**.
- **Reproducibility.** The `runs/` harnesses regenerate every paper number under the pinned environment and
  commit one-line metric files under `reports/`; the `docs/` dashboard links each number to its full
  provenance and a CI-style gate fails if any link breaks.

---

## Repository layout

| Path | Contents |
|------|----------|
| `src/fcl_psp/` | The Python package: `models/` (core pipeline, runtime, perturbation eval, posthoc), `baselines/` (conventional protection baselines), `perturbation/` (measurement-fidelity operators), `evaluation/` (exports, diagnostics). |
| `config/` | Hydra config tree (`main-config.yaml` + groups: `dataset`, `model`, `training`, `window_extraction`, `tracking`, `ablation`, `baseline`, `perturbation`, `data_sparsity`). |
| `runs/` | Regeneration harnesses (`harnesses/`), SLURM launchers + job lists (`drivers/`), and committed per-job console logs (`logs/`). See [`runs/README.md`](runs/README.md). |
| `reports/` | Committed provenance artifacts the dashboard reads: `runs/*.txt`, `baselines/*/cv_summary.json`, `perturbation/*/perturbation_summary.csv`. |
| `docs/` | Results-audit / provenance dashboard (generated static site). See [`docs/README.md`](docs/README.md). `docs/internal/` holds maintainer-only working notes and is not part of the published site. |
| `tests/` | Dataset-free unit tests (label construction, metrics, model factory, baseline/perturbation operators). |
| `notebooks/` | Analysis and figure notebooks; exported vector assets in `notebooks/figures/`. |
| `hpc/` | SLURM scripts for the earlier full-campaign runs. |
| `paper/` | Manuscript LaTeX sources (Elsevier `cas-sc`): `ijepes_2026_framework.tex` (original submission) and `ijepes_2026_framework_revision.tex` (current revision), plus their build dependencies and the `tables/` generators. Built with the tectonic harness `runs/harnesses/compile_redline.py`. |
| `scripts/` | Maintenance scripts — `make_public_copy.sh` builds a scrubbed, history-free export for public release. |

---

## Installation

The reference environment is Python **3.12.4** with `scikit-learn==1.8.0` (hard-pinned — the reported FC/FL
numbers depend on this exact version; do not float it). The framework depends on the companion package
[`psp_helper`](<private-git-host-redacted>), which provides the config
schema, label constants, the Eq. 5 onset-validity rule, and the windowed-data loaders. It is not on PyPI and
requires access to the companion repository.

```bash
# 1) companion package, pinned to the immutable commit the released pipeline runs against
pip install "psp_helper @ git+https://github.com/julianoelhaf/power-grid-and-ai-helper-functions.git@2c78db4"

# 2) this package (+ dev tools for tests and linting)
pip install -e ".[dev]"      # or: make requirements
```

`pyproject.toml` declares `psp_helper` as a pinned git dependency, so step 2 alone pulls it in if your
environment has access. For a byte-for-byte environment, install from the full transitive pin:

```bash
pip install -r requirements-lock.txt   # exact env: Python 3.12.4, scikit-learn 1.8.0, numpy 2.5.1, ...
```

**Optional — 1D-CNN baseline (PyTorch).** The learned CNN baseline runs in a separate conda environment with
`torch` (kept out of the pinned scikit-learn env). It is launched with
`PYTHONPATH=src:runs/harnesses/_shims`, where `_shims/` holds no-op `wandb`/`tabulate` stubs so the framework
imports succeed without those packages. Only needed to reproduce the CNN rows.

---

## Data

Download [PROTECT-90](https://doi.org/10.5281/zenodo.18418330) and point the config at your local copy — either
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

## Results-audit dashboard

`docs/` is a generated static site that ties every reported number to its provenance. Rebuild it from the
committed artifacts (never hand-edit the HTML):

```bash
python3 docs/gen_site_data.py     # reports/*.{txt,json,csv} -> docs/site_data.json (the single number source)
python3 docs/build_pages.py       # manifest.py + site_data.json -> docs/*.html
python3 docs/build_standalone.py  # (optional) flatten to a self-contained docs/standalone.html
python3 docs/verify_links.py      # provenance gate (exits non-zero on any broken link)
```

`verify_links.py` enforces **5-link traceability**: every recorded/derived number must resolve all five links —
the output result file + line where the value appears, the evaluation-code line, the training run-log, the
training-code line, and the config file + Hydra overrides — plus a coverage check that every numeric paper
table is a tracked claim. View `docs/standalone.html` locally, or serve `docs/` as a static site. See
[`docs/README.md`](docs/README.md).

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

CI runs lint + tests via `.gitlab-ci.yml`. The dashboard provenance gate (`docs/verify_links.py`) is run
manually / as a documentation check.

---

## Citation

The framework paper is **under review** at the Elsevier *International Journal of Electrical Power & Energy
Systems* (manuscript IJEPES-D-26-01815); the citation will be finalized on publication. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff).

If you use this code, please cite the **PROTECT-90 dataset**:

```bibtex
@misc{kordowich2026protect90,
  author    = {Kordowich, Georg and Oelhaf, Julian and Bergler, Christian and
               Maier, Andreas and Bayer, Siming and J{\"a}ger, Johann},
  title     = {{PROTECT-90}: A Fault Dataset for Power System Protection},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.0},
  doi       = {10.5281/zenodo.18418330},
  url       = {https://doi.org/10.5281/zenodo.18418330},
  note      = {Dataset. Concept DOI (all versions): 10.5281/zenodo.18418329}
}
```

Funding: Deutsche Forschungsgemeinschaft (DFG) — 535389056.

---

## Contact

Julian Oelhaf — [julian.oelhaf@fau.de](mailto:julian.oelhaf@fau.de) ·
[Website](https://lme.tf.fau.de/persons/julian-oelhaf/) ·
[<private-git-host-redacted>/anonymized-user](<private-git-host-redacted>)

## License

BSD-3-Clause. See [LICENSE](LICENSE).
