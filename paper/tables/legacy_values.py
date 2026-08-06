"""Cells with NO committed regeneration in ``reports/`` (as of 2026-07-17).

These are transcribed verbatim from ``paper/ijepe_revision.tex`` -- they are the
disclosed original-run / W&B values (see the reproducibility note at
ijepe_revision.tex:644) and the not-yet-regenerated hyperparameter-ablation cells
(open item: independent MLP ablation regeneration). Any table generator that reads
from here also emits a WARNING via ``sources.warn`` so the provenance report flags
exactly which cells are not committed-backed.

If/when these runs are committed, delete the relevant block and switch the
generator to read from ``reports/``.
"""

# --- tab:fl_timing_sensitivity, 30 & 40 ms (no committed FL 30/40 ms run) ---
# repro_fl.txt only has W=0.020 and W=0.050.  (mean, std)
FL_TIMING = {
    ("30", "mlp"): (10.18, 0.35), ("30", "gb"): (14.64, 0.18),
    ("30", "knn"): (19.65, 0.19), ("30", "ridge"): (26.11, 0.35),
    ("40", "mlp"): (10.46, 0.52), ("40", "gb"): (14.59, 0.19),
    ("40", "knn"): (19.33, 0.16), ("40", "ridge"): (26.12, 0.32),
}

# --- tab:fc_timing_sensitivity, 30 & 40 ms, KNN + Ridge only ---
# run_fc_results.txt has tim_{mlp,gb}_W{30,40} but no KNN/Ridge at 30/40 ms.  (mean, std)
FC_TIMING = {
    ("30", "knn"): (0.831, 0.004), ("30", "ridge"): (0.093, 0.002),
    ("40", "knn"): (0.851, 0.003), ("40", "ridge"): (0.091, 0.002),
}

# The two ablation tables are no longer legacy: the 108-run campaign (reports/runs/run_ablation_results.txt,
# job 774769) is committed, and table_ablation_*.py now compute from it via sources.ablation_agg.
