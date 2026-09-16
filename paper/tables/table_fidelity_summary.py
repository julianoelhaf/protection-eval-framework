"""tab:fidelity_summary -- compact measurement-fidelity summary at 20 ms.

Clean = the five-fold reference statistic mean (repro_fc/fl). Worst case = the single
most-degraded perturbed level across the three axes (max MAE for FL, min macro-F1 for FC),
from perturbation_summary.csv, excluding the clean-equivalent pseudo-levels. Focused MLP/GB
panel. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

PSEUDO_CLEAN = {("noise", "clean"), ("jitter", "0"), ("ct_saturation", "no_sat")}


def worst(name, fl):
    """Worst-case mean over real perturbed levels: max MAE (FL) or min macro-F1 (FC)."""
    pert = S.load_perturbation(name)
    vals = [v["mean"] for k, v in pert.items() if k not in PSEUDO_CLEAN]
    return max(vals) if fl else min(vals)


def build():
    rfc = S.parse_repro(S.RUNS / "repro_fc.txt")
    rfl = S.parse_repro(S.RUNS / "repro_fl.txt")

    fl_lbl = "\\ac{fl} (\\ac{mae} [\\%], $\\downarrow$)"
    fc_lbl = "\\ac{fc} (macro-\\ac{f1}, $\\uparrow$)"
    rows = [
        (fl_lbl, "\\ac{mlp}", rfl[("y_fault_location", "mlp_regressor", "W=0.020")]["mean"],
         worst("mlp_regressor_y_fault_location_W20ms", True), 2),
        (fl_lbl, "\\ac{gb} ", rfl[("y_fault_location", "hist_gradient_boosting_regressor", "W=0.020")]["mean"],
         worst("hist_gradient_boosting_regressor_y_fault_location_W20ms", True), 2),
        None,
        (fc_lbl, "\\ac{mlp}", rfc[("event_type", "mlp_classifier", "W=0.020")]["mean"],
         worst("mlp_classifier_event_type_W20ms", False), 3),
        (fc_lbl, "\\ac{gb} ", rfc[("event_type", "hist_gradient_boosting_classifier", "W=0.020")]["mean"],
         worst("hist_gradient_boosting_classifier_event_type_W20ms", False), 3),
    ]

    body = []
    for r in rows:
        if r is None:
            body.append("\\midrule")
            continue
        lbl, model, clean, wc, dp = r
        body.append("%s & %s & %s & %s \\\\" % (lbl, model, S.fnum(clean, dp), S.fnum(wc, dp)))

    latex = r"""\begin{table}[pos=t]
\caption{Compact measurement-fidelity summary at 20\,ms for the focused sensitivity panel: clean value and worst-case level across the three degradation axes (additive noise, current-transformer saturation, synchronization jitter). Full per-level results in Appendix~\ref{app:fidelity} (Tables~\ref{tab:fidelity_fl} and~\ref{tab:fidelity_fc}).}
\label{tab:fidelity_summary}
\small
\centering
\begin{revblock}
\begin{tabular}{llcc}
\toprule
\textbf{Task (metric)} & \textbf{Model} & \textbf{Clean} & \textbf{Worst case} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{revblock}
\end{table}""" % "\n".join(body)

    S.write_table("fidelity_summary", latex,
                  "repro clean + perturbation_summary.csv worst-case (MLP/GB, W20) (committed)")


if __name__ == "__main__":
    build()
