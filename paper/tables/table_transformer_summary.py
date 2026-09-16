"""tab:transformer_summary -- main-text learned-baseline comparison (MLP / CNN1D).

Sources: repro_fc/fl (MLP reference) + reports/runs/cnn_results.txt (CNN1D from scratch).
FC = macro-F1 (higher better), FL = MAE % (lower better). Values are mean $\\pm$ std over the
five held-out folds; best mean per column bolded. The pre-trained MOMENT-1-large baseline is
reported separately in Appendix (see table_transformer_appendix). All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

COLS = [("fc", "20", 3, "max"), ("fc", "50", 3, "max"),
        ("fl", "20", 2, "min"), ("fl", "50", 2, "min")]


def mlp_ref(rfc, rfl, t, w):
    if t == "fc":
        return rfc[("event_type", "mlp_classifier", "W=0.0%s" % w)]
    return rfl[("y_fault_location", "mlp_regressor", "W=0.0%s" % w)]


def cell(v, dp, is_best, is_fl):
    s = S.pm(v["mean"], v["std"], dp)
    if is_fl and v["mean"] < 10:
        s = "\\phantom{0}" + s
    return "\\textbf{%s}" % s if is_best else s


def build():
    rfc = S.parse_repro(S.RUNS / "repro_fc.txt")
    rfl = S.parse_repro(S.RUNS / "repro_fl.txt")
    cnn = S.parse_labeled(S.RUNS / "cnn_results.txt")

    mlp = {(t, w): mlp_ref(rfc, rfl, t, w) for t, w, _, _ in COLS}
    cnnv = {(t, w): cnn["cnn_%s_W%s" % (t, w)] for t, w, _, _ in COLS}

    rows = [("\\ac{mlp} (reference)", mlp), ("\\ac{cnn1d} (from scratch)", cnnv)]
    best = {}
    for t, w, _, mode in COLS:
        vals = {name: d[(t, w)]["mean"] for name, d in rows}
        best[(t, w)] = (max if mode == "max" else min)(vals, key=vals.get)

    body = []
    for name, d in rows:
        cells = [cell(d[(t, w)], dp, best[(t, w)] == name, t == "fl") for t, w, dp, _ in COLS]
        body.append("%-26s & %s \\\\" % (name, " & ".join(cells)))

    latex = r"""\begin{table}[pos=t]
\caption{Main-text learned-baseline comparison under the shared episode-grouped five-fold protocol. \Ac{fc} is evaluated using macro-\ac{f1} ($\uparrow$), and \ac{fl} using \ac{mae} in percent of line length ($\downarrow$). Values are mean $\pm$ standard deviation across the five held-out folds; the best mean in each column is shown in bold. The pre-trained MOMENT-1-large results are reported separately in Appendix~\ref{app:transformer} (Table~\ref{tab:transformer_appendix}).}
\label{tab:transformer_summary}
\small
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{\ac{fc} 20\,ms} & \textbf{\ac{fc} 50\,ms} & \textbf{\ac{fl} 20\,ms} & \textbf{\ac{fl} 50\,ms} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("transformer_summary", latex, "repro (MLP) + cnn_results (committed)")


if __name__ == "__main__":
    build()
