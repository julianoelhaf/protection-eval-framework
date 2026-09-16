"""tab:fc_runtime_compact -- runtime for FC (reference assumptions).

Runtime source: reports/runs/run_runtime_results.txt (released-code timing on the
recorded SLURM node). F1 context: repro_fc.txt. Per window group, MLP's F1 (max) and
inference (min) are bolded. The caption records the actual node (these are
machine-specific indicators). All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FC_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier"}
GROUPS = [("20", ["mlp", "gb"]), ("50", ["mlp", "gb"])]


def build():
    rt = S.parse_runtime(S.RUNS / "run_runtime_results.txt")
    fc = S.parse_repro(S.RUNS / "repro_fc.txt")
    node = rt["rt_fc_mlp_W20"]

    body = []
    for gi, (w, models) in enumerate(GROUPS):
        if gi:
            body.append("\\midrule")
        f1 = {m: fc[("event_type", FC_KEY[m], "W=0.0%s" % w)]["mean"] for m in models}
        rr = {m: rt["rt_fc_%s_W%s" % (m, w)] for m in models}
        best_f1 = max(models, key=lambda m: f1[m])
        best_inf = min(models, key=lambda m: rr[m]["infer"])
        for m in models:
            f1s = S.fnum(f1[m], 3)
            infs = S.fnum(rr[m]["infer"], 2)
            body.append("\\ac{%s} & %s\\,ms & %s & %s & %s & %s \\\\" % (
                m, w,
                "\\textbf{%s}" % f1s if m == best_f1 else f1s,
                S.fnum(rr[m]["train"], 1),
                "\\textbf{%s}" % infs if m == best_inf else infs,
                S.fnum(rr[m]["thr"], 1)))

    latex = r"""\begin{table}[ht]
\caption{Compact runtime summary for fault classification under the reference assumptions. Macro-\ac{f1} is shown for context. Runtimes are measured on a %d-core SLURM node (%s, OMP=%d) and are machine-specific relative indicators, not end-to-end relay latencies.}
\label{tab:fc_runtime_compact}
\centering
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Window} & \textbf{F1} & \textbf{Train [s]} & \textbf{Infer. [$\mu$s]} & \textbf{Thr. [k/s]} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % (node["cpus"], node["node"], node["omp"], "\n".join(body))

    S.write_table("fc_runtime_compact", latex,
                  "run_runtime (node %s) + repro_fc F1 (committed)" % node["node"])


if __name__ == "__main__":
    build()
