# Paper table generators

Each numeric table in `paper/ijepe_revision.tex` has a generator here that **extracts the
numbers from the committed evidence in `reports/`** and writes a drop-in
`table__<name>.tex` (a full `table` float, matching the manuscript's caption/label/format).
Nothing is hand-typed except the clearly-labelled cells in `legacy_values.py`.

## Usage

```bash
python paper/tables/build_all.py            # regenerate all 19 tables + print provenance
python paper/tables/table_reference_20ms.py # regenerate one table
```

Each `table__<name>.tex` is self-contained: `\input{tables/table__reference_20ms}` (or paste
the float). Tables that sit inside a section-level `revblock` in the manuscript
(`baselines_fl`, `baselines_obs`, `fidelity_fl`, `fidelity_fc`) reproduce their own
`revblock`; `generalization_shift` does **not** (its `revblock` is at the prose level).

## Files

| Module | Table | Source(s) |
|---|---|---|
| `sources.py` | — | shared parsers + LaTeX formatters (ddof=0 std) |
| `legacy_values.py` | — | cells with **no** committed run (transcribed from the manuscript) |
| `table_reference_20ms` | `tab:reference_20ms` | `repro_fc/fl` W=0.020 |
| `table_fc_timing_sensitivity` | `tab:fc_timing_sensitivity` | `run_tim10` + `repro_fc` + `run_fc` `tim_*`; 30/40 ms KNN+Ridge legacy |
| `table_fl_timing_sensitivity` | `tab:fl_timing_sensitivity` | `run_tim10` + `repro_fl`; 30/40 ms legacy |
| `table_fc_observability` | `tab:fc_observability` | `run_obs` aggregate + `repro_fc` (Full) |
| `table_fl_observability` | `tab:fl_observability` | `run_obs` aggregate + `repro_fl` (Full) |
| `table_baselines_fl` | `tab:baselines_fl` | `repro_fl` + `baselines/*_ended` `cv_summary.json` |
| `table_baselines_obs` | `tab:baselines_obs` | `repro_fl` + `run_obs` + baselines settled |
| `table_fidelity_fl` | `tab:fidelity_fl` | `perturbation_summary.csv` (FL MLP+GB W20) |
| `table_fidelity_fc` | `tab:fidelity_fc` | `perturbation_summary.csv` (FC MLP+GB W20) |
| `table_generalization_shift` | `tab:generalization_shift` | `repro` (in-dist) + `b4_generalization_results` (single-draw) |
| `table_fc_runtime_compact` | `tab:fc_runtime_compact` | `run_runtime` (node lme222) + `repro_fc` |
| `table_fl_runtime_compact` | `tab:fl_runtime_compact` | `run_runtime` (node lme222) + `repro_fl` |
| `table_fc_stride_sensitivity` | `tab:fc_stride_sensitivity` | `run_stride_results.txt` |
| `table_fl_stride_sensitivity` | `tab:fl_stride_sensitivity` | `run_stride_results.txt` |
| `table_fl_observability_detail_mlp_50ms` | `tab:fl_observability_detail_mlp_50ms` (App. A1) | `run_obs` `obs_fl_mlp_{s,p}*_W50` |
| `table_ablation_robustness_summary_main` | `tab:ablation_robustness_summary_main` | **legacy** (open item) |
| `table_ablation_robustness_condensed` | `tab:ablation_robustness_condensed` | **legacy** (open item) |
| `table_transformer_summary` | `tab:transformer_summary` (new; learned-baseline probe) | `repro` (MLP) + `cnn_results.txt` + `moment_results.txt` |
| `table_class_distribution` | `tab:class_distribution` (new; App.) | `class_split_stats.txt` |

Bolding is computed by rule: best model per row (max macro-F1 / min MAE) for the
reference/timing/runtime tables; best per column for the observability tables; the fidelity
"collapse" cell and the two-ended-settled baseline are bolded as in the manuscript.

## Provenance / conventions

- **Std is population std (ddof=0)**, matching the harnesses (verified against per-fold values).
- **Observability aggregates:** Full = the 20/50 ms reference; *Relay pair* = mean over the 4
  same-line pairs; *Single relay* = mean over the 8 single-relay runs.
- **Generated values match the gate-verified dashboard** (`docs/`), which records the committed
  run values. The manuscript's typed values are tracked separately there as "paper values" with
  small drifts.

## Differences vs. the current manuscript (review before adopting)

Generated = committed evidence, so a few cells differ from `ijepe_revision.tex`. **These are
expected**; the author decides which to adopt.

**Mean (headline) — one decision:**
- **FC MLP · 20 ms macro-F1: generated `0.991` vs manuscript `0.990`.** `repro_fc` mean is
  `0.9908`, which rounds up to `0.991` at 3 dp (this is exactly what the dashboard records; the
  `0.990` in the paper is the within-tolerance "paper value", drift +0.0008). Appears in
  `reference_20ms`, `fc_timing` (20 ms), `fc_observability` (Full 20 ms), `fc_runtime` (20 ms),
  `generalization_shift` (in-dist). **Keep `0.990` as the canonical headline, or adopt `0.991`
  consistently.** (MLP-FC · 50 ms is `0.9901` → `0.990`, unchanged.)

**Std corrections (all from committed runs):**
- MLP-FL: `0.37 → 0.25` (20 ms), `0.33 → 0.27` (50 ms).
- MLP-FC · 50 ms: `0.003 → 0.005`. MLP-FC · 20 ms: `0.002 → 0.001`. GB-FC · 20 ms: `0.019 → 0.017`.
  KNN-FL · 20 ms: `0.17 → 0.16`.
- `fidelity_fl` clean row: `0.28 / 0.28` (perturbation-protocol std over folds×realizations; the
  manuscript had `0.37 / 0.25`).

**Adopted committed values that differ from the manuscript's disclosed original-run cells:**
- 10 ms rows: MLP-FC `0.989±0.003 → 0.985±0.011`; MLP-FL `10.66±0.39 → 10.64±0.36`.
  *(If adopting, update the reproducibility note at `ijepe_revision.tex:644`, which currently
  excludes 10 ms.)*
- Reduced-observability aggregates: ≤0.01 shifts (e.g. GB-FC pair 20 ms `0.760→0.750`; MLP-FC pair
  `0.987→0.989`; MLP-FL pair 20 ms `19.59→19.65`, 50 ms `19.56→19.62`; MLP-FL single 50 ms
  `22.15→22.17`). *(Same L644 caveat.)*
- Stride Δ cells: regenerated (see the cell-diffs doc §2B); FL-GB rows unchanged, MLP + FC-GB shift.
- Runtime: lme222 figures (machine-specific); the caption gains a node-disclosure sentence.
- App-A1: regenerated per-line values (drift ≤0.13 %).
- Bolding: `fc_timing` 40 ms bolds MLP (row-max) rather than the manuscript's GB (a likely typo).

**Not regenerated (WARNING on build):**
- `fl_timing` 30 & 40 ms (no committed FL 30/40 ms run) → `legacy_values.FL_TIMING`.
- `fc_timing` 30 & 40 ms KNN + Ridge (no committed run) → `legacy_values.FC_TIMING`.
- Both ablation tables (only FC-20 ms `abl_*` is committed; the FL and 50 ms cells are the open
  regeneration item) → `legacy_values.ABLATION_*`. Rewire once that campaign lands.

`generalization_shift` uses the single-draw committed values (matches the current table). A
seed-mean ± std variant for the MLP rows is also committed — see the cell-diffs doc, §2C.
