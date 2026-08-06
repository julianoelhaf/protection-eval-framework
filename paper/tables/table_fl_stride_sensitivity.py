"""tab:fl_stride_sensitivity -- MAE vs stride (FL).

Baseline = the reference (repro_fl) 5 ms-stride evaluation for each (model, window) where it
exists (20/50 ms); the 10 ms window is not part of the reference sweep, so its baseline is the
stride-campaign 5 ms run. Delta @ k = MAE(stride=k) - baseline. Negative = improvement.
Sources: reports/runs/run_stride_results.txt + reports/runs/repro_fl.txt. Missing combinations
render as '--'. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

MODELS = ["mlp", "gb"]
WINDOWS = ["10", "20", "50"]
STRIDES = ["10", "20", "50"]
REPRO_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor"}


def build():
    st = S.parse_bracket(S.RUNS / "run_stride_results.txt")
    repro = S.parse_repro(S.RUNS / "repro_fl.txt")

    def val(model, w, s):
        n = st.get("stride_fl_%s_W%s_S%s" % (model, w, s))
        return None if n is None else n["mean"]

    def baseline(model, w):
        key = ("y_fault_location", REPRO_KEY[model], "W=0.0%s" % w)
        if key in repro:
            return repro[key]["mean"]
        return val(model, w, "5")  # 10 ms window: reference sweep has no 10 ms point

    body = []
    for mi, model in enumerate(MODELS):
        if mi:
            body.append("\\midrule")
        for w in WINDOWS:
            base = baseline(model, w)
            cells = [S.fnum(base, 2)]
            for k in STRIDES:
                v = val(model, w, k)
                cells.append(S.signed(v - base, 2) if v is not None else "--")
            body.append("\\ac{%s} & %s\\,ms & %s \\\\" % (model, w, " & ".join(cells)))

    latex = r"""\begin{table}[pos=ht]
\caption{Compact stride sensitivity for fault localization. Baseline performance is reported at the default 5\,ms stride; additional columns show $\Delta$\ac{mae} relative to that baseline for larger strides. Negative values indicate improvement.}
\small
\label{tab:fl_stride_sensitivity}
\centering
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Window} & \textbf{Baseline \acs{mae}} & \textbf{$\Delta$ @ 10\,ms} & \textbf{$\Delta$ @ 20\,ms} & \textbf{$\Delta$ @ 50\,ms} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("fl_stride_sensitivity", latex,
                  "run_stride_results.txt + repro_fl reference baseline (committed)")


if __name__ == "__main__":
    build()
