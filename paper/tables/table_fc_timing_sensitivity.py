"""tab:fc_timing_sensitivity -- macro-F1 across decision horizons (FC).

Sources: 10 ms = run_tim10_results.txt; 20/50 ms = repro_fc.txt; 30/40 ms MLP+GB =
run_fc_results.txt (tim_*). 30/40 ms KNN+Ridge have NO committed run -> legacy_values.
Best (max) model per row is bolded.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S
import legacy_values as L

MODELS = ["mlp", "gb", "knn", "ridge"]
FC_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier",
          "knn": "k_neighbors_classifier", "ridge": "ridge_classifier"}


def build():
    repro = S.parse_repro(S.RUNS / "repro_fc.txt")
    runfc = S.parse_bracket(S.RUNS / "run_fc_results.txt")
    tim10 = S.parse_bracket(S.RUNS / "run_tim10_results.txt")

    def val(w, m):
        if w == "10":
            n = tim10["tim10_fc_%s" % m]
            return n["mean"], n["std"]
        if w in ("20", "50"):
            n = repro[("event_type", FC_KEY[m], "W=0.0%s" % w)]
            return n["mean"], n["std"]
        # 30 / 40 ms
        if m in ("mlp", "gb"):
            n = runfc["tim_%s_W%s" % (m, w)]
            return n["mean"], n["std"]
        S.warn("tab:fc_timing_sensitivity: %s ms %s from legacy_values (no committed run)" % (w, m.upper()))
        return L.FC_TIMING[(w, m)]

    rows_meta = [("10", "10 ms"), ("20", "20 ms (ref.)"), ("30", "30 ms"),
                 ("40", "40 ms"), ("50", "50 ms")]
    body = []
    for w, label in rows_meta:
        vals = {m: val(w, m) for m in MODELS}
        best = max(MODELS, key=lambda m: vals[m][0])
        cells = []
        for m in MODELS:
            s = S.pm(vals[m][0], vals[m][1], 3)
            cells.append("\\textbf{%s}" % s if m == best else s)
        body.append("%s & %s \\\\" % (label, " & ".join(cells)))

    latex = r"""\begin{table}[ht]
\caption{Controlled timing sensitivity for fault classification. Macro-\ac{f1} across decision horizons for the representative case-study models. Mean $\pm$ std over 5 folds; higher is better. The 20\,ms row corresponds to the fixed reference timing configuration.}
\small
\label{tab:fc_timing_sensitivity}
\centering
\begin{tabular}{lcccc}
\toprule
\bfseries Window & \bfseries \ac{mlp} & \bfseries \ac{gb} & \bfseries \ac{knn} & \bfseries \ac{ridge} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("fc_timing_sensitivity", latex,
                  "tim10 + repro_fc + run_fc tim_* (committed); 30/40 ms KNN+Ridge legacy")


if __name__ == "__main__":
    build()
