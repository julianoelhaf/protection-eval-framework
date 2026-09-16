"""tab:reference_20ms -- reference performance at the fixed 20 ms configuration.

Source: reports/runs/repro_fc.txt (FC macro-F1) + repro_fl.txt (FL MAE), W=0.020.
Best model per row is bolded (max macro-F1 for FC, min MAE for FL).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

MODELS = ["mlp", "gb", "knn", "ridge"]
FC_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier",
          "knn": "k_neighbors_classifier", "ridge": "ridge_classifier"}
FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor",
          "knn": "k_neighbors_regressor", "ridge": "ridge_regressor"}


def build():
    fc = S.parse_repro(S.RUNS / "repro_fc.txt")
    fl = S.parse_repro(S.RUNS / "repro_fl.txt")

    fc_v = {m: fc[("event_type", FC_KEY[m], "W=0.020")] for m in MODELS}
    fl_v = {m: fl[("y_fault_location", FL_KEY[m], "W=0.020")] for m in MODELS}

    fc_best = max(MODELS, key=lambda m: fc_v[m]["mean"])
    fl_best = min(MODELS, key=lambda m: fl_v[m]["mean"])

    def cell(v, dp, best):
        s = S.pm(v["mean"], v["std"], dp)
        return "\\textbf{%s}" % s if best else s

    fc_cells = " & ".join(cell(fc_v[m], 3, m == fc_best) for m in MODELS)
    fl_cells = " & ".join(cell(fl_v[m], 2, m == fl_best) for m in MODELS)

    latex = r"""\begin{table}[ht]
\caption{Reference performance across representative case-study models for the fixed 20\,ms timing configuration. \ac{fc} is reported by macro-\ac{f1}; \ac{fl} by \ac{mae} in percent of normalized line length. Values are mean $\pm$ std over 5 folds.}
\small
\label{tab:reference_20ms}
\centering
\begin{tabular}{llcccc}
\toprule
\bfseries Task & \bfseries Metric & \bfseries \acs{mlp} & \bfseries \ac{gb} & \bfseries \ac{knn} & \bfseries Ridge \\
\midrule
\ac{fc} & Macro-\ac{f1} $\uparrow$ & %s \\
\ac{fl} & \ac{mae} [\%%] $\downarrow$ & %s \\
\bottomrule
\end{tabular}
\end{table}""" % (fc_cells, fl_cells)

    S.write_table("reference_20ms", latex, "repro_fc/fl W=0.020 (committed)")


if __name__ == "__main__":
    build()
