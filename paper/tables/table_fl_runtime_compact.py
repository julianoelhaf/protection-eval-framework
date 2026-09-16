"""tab:fl_runtime_compact -- runtime for FL (reference assumptions).

Runtime source: reports/runs/run_runtime_results.txt. MAE context: repro_fl.txt.
Per window group, MLP's MAE (min) and inference (min) are bolded. Caption records
the actual node (machine-specific indicators). All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor"}
GROUPS = [("20", ["mlp", "gb"]), ("50", ["mlp", "gb"])]


def build():
    rt = S.parse_runtime(S.RUNS / "run_runtime_results.txt")
    fl = S.parse_repro(S.RUNS / "repro_fl.txt")
    node = rt["rt_fl_mlp_W20"]

    body = []
    for gi, (w, models) in enumerate(GROUPS):
        if gi:
            body.append("\\midrule")
        mae = {m: fl[("y_fault_location", FL_KEY[m], "W=0.0%s" % w)]["mean"] for m in models}
        rr = {m: rt["rt_fl_%s_W%s" % (m, w)] for m in models}
        best_mae = min(models, key=lambda m: mae[m])
        best_inf = min(models, key=lambda m: rr[m]["infer"])
        for m in models:
            maes = S.fnum(mae[m], 2)
            infs = S.fnum(rr[m]["infer"], 2)
            body.append("\\ac{%s} & %s\\,ms & %s & %s & %s & %s \\\\" % (
                m, w,
                "\\textbf{%s}" % maes if m == best_mae else maes,
                S.fnum(rr[m]["train"], 1),
                "\\textbf{%s}" % infs if m == best_inf else infs,
                S.fnum(rr[m]["thr"], 1)))

    latex = r"""\begin{table}[ht]
\caption{Compact runtime summary for fault localization under the reference assumptions. \ac{mae} [\%%] is shown for context. Runtimes are measured on a %d-core SLURM node (%s, OMP=%d) and are machine-specific relative indicators, not end-to-end relay latencies.}
\label{tab:fl_runtime_compact}
\centering
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{Window} & \textbf{MAE} & \textbf{Train [s]} & \textbf{Infer. [$\mu$s]} & \textbf{Thr. [k/s]} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % (node["cpus"], node["node"], node["omp"], "\n".join(body))

    S.write_table("fl_runtime_compact", latex,
                  "run_runtime (node %s) + repro_fl MAE (committed)" % node["node"])


if __name__ == "__main__":
    build()
