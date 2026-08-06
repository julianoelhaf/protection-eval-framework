"""tab:fc_stride_sensitivity -- macro-F1 vs stride (FC).

Baseline = the reference (repro_fc) 5 ms-stride evaluation for each (model, window) where it
exists (20/50 ms); the 10 ms window is not part of the reference sweep, so its baseline is the
stride-campaign 5 ms run. Delta @ k = macroF1(stride=k) - baseline. Positive = improvement.
Sources: reports/runs/run_stride_results.txt + reports/runs/repro_fc.txt. Missing combinations
render as '--'. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

MODELS = ["mlp", "gb"]
WINDOWS = ["10", "20", "50"]
STRIDES = ["10", "20", "50"]
REPRO_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier"}


def build():
    st = S.parse_bracket(S.RUNS / "run_stride_results.txt")
    repro = S.parse_repro(S.RUNS / "repro_fc.txt")

    def val(model, w, s):
        n = st.get("stride_fc_%s_W%s_S%s" % (model, w, s))
        return None if n is None else n["mean"]

    def baseline(model, w):
        key = ("event_type", REPRO_KEY[model], "W=0.0%s" % w)
        if key in repro:
            return repro[key]["mean"]
        return val(model, w, "5")  # 10 ms window: reference sweep has no 10 ms point

    body = []
    for mi, model in enumerate(MODELS):
        if mi:
            body.append("\\midrule")
        for w in WINDOWS:
            base = baseline(model, w)
            cells = [S.fnum(base, 3)]
            for k in STRIDES:
                v = val(model, w, k)
                cells.append(S.signed(v - base, 3) if v is not None else "--")
            body.append("\\ac{%s} & %s\\,ms & %s \\\\" % (model, w, " & ".join(cells)))

    latex = r"""\begin{table}[pos=ht]
\caption{Compact stride sensitivity for fault classification. Baseline performance is reported at the default 5\,ms stride; additional columns show $\Delta$ macro-\ac{f1} relative to that baseline for larger strides. Positive values indicate improvement.}
\small
\label{tab:fc_stride_sensitivity}
\centering
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Window} & \textbf{Baseline \acs{f1}} & \textbf{$\Delta$ @ 10\,ms} & \textbf{$\Delta$ @ 20\,ms} & \textbf{$\Delta$ @ 50\,ms} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("fc_stride_sensitivity", latex,
                  "run_stride_results.txt + repro_fc reference baseline (committed)")


if __name__ == "__main__":
    build()
