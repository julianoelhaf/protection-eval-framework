# Paper table generators

Each numeric table in the manuscript has a generator here that **extracts the numbers from the
committed evidence in `reports/`** and writes a drop-in `table__<name>.tex` (a full `table` float,
matching the manuscript's caption/label/format). Nothing is hand-typed except the clearly-labelled
cells in `legacy_values.py`.

## Usage

```bash
python paper/tables/build_all.py            # regenerate all 21 tables + print provenance
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
| `table_ablation_robustness_summary_main` | `tab:ablation_robustness_summary_main` | `run_ablation_results.txt` (108-run campaign) via `sources.ablation_agg`, over both horizons |
| `table_ablation_robustness_condensed` | `tab:ablation_robustness_condensed` | `run_ablation_results.txt`, per-window via `sources.ablation_agg` |
| `table_transformer_summary` | `tab:transformer_summary` (main text; MLP / CNN1D) | `repro` (MLP) + `cnn_results.txt` |
| `table_transformer_appendix` | `tab:transformer_appendix` (App.; + MOMENT probe/head) | `repro` (MLP) + `cnn_results.txt` + `moment_results.txt` |
| `table_fidelity_summary` | `tab:fidelity_summary` | `perturbation_summary.csv` (clean + worst case) |
| `table_class_distribution` | `tab:class_distribution` (App.) | `class_split_stats.txt` |

Bolding is computed by rule: best model per row (max macro-F1 / min MAE) for the
reference/timing/runtime tables; best per column for the observability tables; the fidelity
"collapse" cell and the two-ended-settled baseline are bolded as in the manuscript.

## Provenance / conventions

- **Std is population std (ddof=0)**, matching the harnesses (verified against per-fold values).
- **Observability aggregates:** Full = the 20/50 ms reference, reported as mean ± std over the 5
  episode-grouped folds; *Relay pair* = mean over the 4 same-line pairs and *Single relay* = mean
  over the 8 single-relay runs, both reported with the **min–max range** across those configs in
  brackets. The two dispersions are different quantities: a fold std on the reduced rows would
  understate sensitivity to relay choice, which is the scientifically meaningful spread there.
- **Generated values are the committed run values** recorded under `reports/`.

## Agreement with the manuscript

**All 21 tables reproduce the published manuscript cell-for-cell** on the numeric tabular bodies.
Every value these generators emit is therefore the committed-evidence value, and matches what the
paper prints.

Getting there resolved the drifts this section used to enumerate. Two worth remembering, because
they set the precedent for the next disagreement:

- **The manuscript moved, not the generator.** Three ≤0.01 cells were manuscript hand-rounding
  artifacts (`fl_stride` MLP-50 `+1.09/+1.43` → `+1.10/+1.44`; `obs_detail` 02–03A `18.798` →
  `18.797` at a `.xxx5` boundary; `ablation_condensed` FL MLP-20 `+0.676` → `+0.675`). They were
  corrected in the manuscript, **not** hardcoded into the generators — the generator is the
  faithful reproduction of committed evidence.
- **FC MLP · 20 ms macro-F1 adopted `0.991`** (`repro_fc` mean `0.9908`, rounds up at 3 dp) in place
  of the earlier `0.990` paper value, consistently across `reference_20ms`, `fc_timing`,
  `fc_observability`, `fc_runtime` and `generalization_shift`.

If a future cell disagrees, fix the manuscript or the evidence — do not hand-patch a generator.

## Not regenerated (WARNING on build)

Only the timing tables still read hand-transcribed cells; everything else is committed-backed.

- `fl_timing` 30 & 40 ms (no committed FL 30/40 ms run) → `legacy_values.FL_TIMING`.
- `fc_timing` 30 & 40 ms KNN + Ridge (no committed run) → `legacy_values.FC_TIMING`.

The two ablation tables are **no longer legacy**: the 108-run ablation campaign
(`run_ablation_results.txt`) is committed and both generators compute from it via
`sources.ablation_agg`.

`generalization_shift` uses the single-draw committed values for GB (seed 42) and, for the shifted
MLP cells, the mean ± population std over the five `b4_generalization_seeds` runs.
