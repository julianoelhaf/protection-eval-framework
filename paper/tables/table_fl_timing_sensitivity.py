"""tab:fl_timing_sensitivity -- MAE (% line) across decision horizons (FL).

Sources: 10 ms = run_tim10_results.txt; 20/50 ms = repro_fl.txt. 30/40 ms have NO
committed FL run -> legacy_values.FL_TIMING. Best (min) model per row is bolded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S
import legacy_values as L

MODELS = ["mlp", "gb", "knn", "ridge"]
FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor",
          "knn": "k_neighbors_regressor", "ridge": "ridge_regressor"}


def build():
    repro = S.parse_repro(S.RUNS / "repro_fl.txt")
    tim10 = S.parse_bracket(S.RUNS / "run_tim10_results.txt")

    def val(w, m):
        if w == "10":
            n = tim10["tim10_fl_%s" % m]
            return n["mean"], n["std"]
        if w in ("20", "50"):
            n = repro[("y_fault_location", FL_KEY[m], "W=0.0%s" % w)]
            return n["mean"], n["std"]
        S.warn("tab:fl_timing_sensitivity: %s ms %s from legacy_values (no committed run)" % (w, m.upper()))
        return L.FL_TIMING[(w, m)]

    rows_meta = [("10", "10 ms"), ("20", "20 ms (ref.)"), ("30", "30 ms"),
                 ("40", "40 ms"), ("50", "50 ms")]
    body = []
    for w, label in rows_meta:
        vals = {m: val(w, m) for m in MODELS}
        best = min(MODELS, key=lambda m: vals[m][0])
        cells = []
        for m in MODELS:
            s = S.pm(vals[m][0], vals[m][1], 2)
            cells.append("\\textbf{%s}" % s if m == best else s)
        body.append("%s & %s \\\\" % (label, " & ".join(cells)))

    latex = r"""\begin{table}[ht]
\caption{Timing sensitivity for \ac{fl}. \ac{mae} is reported as percent of normalized line length across decision horizons. Values are mean $\pm$ std over 5 folds; lower is better. The 20\,ms setting is the reference configuration.}
\small
\label{tab:fl_timing_sensitivity}
\centering
\begin{tabular}{lcccc}
\toprule
\bfseries Window & \bfseries \ac{mlp} & \bfseries \ac{gb} & \bfseries \ac{knn} & \bfseries \ac{ridge} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("fl_timing_sensitivity", latex,
                  "tim10 + repro_fl (committed); 30/40 ms all models legacy")


if __name__ == "__main__":
    build()
