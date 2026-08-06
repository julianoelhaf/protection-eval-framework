"""tab:ablation_robustness_summary_main -- compact hyperparameter robustness summary.

Source: reports/runs/run_ablation_results.txt (committed 108-run campaign, job 774769); aggregation in
sources.ablation_agg over both decision horizons. Ablation default = best tagged default among horizons;
Best = strongest ablation result; Delta_best = the improvement magnitude (increase in macro-F1 for FC,
reduction in MAE for FL) -- positive means improvement in both tasks, matching the condensed table and
the manuscript; Max. spread = largest best-worst within one single-parameter group. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

TASK_AC = {"fc": "\\ac{fc}", "fl": "\\ac{fl}"}
MODEL_AC = {"gb": "\\ac{gb}", "mlp": "\\ac{mlp}"}
# paper row order: MLP-FC, GB-FC, then MLP-FL, GB-FL
ORDER = [("mlp", "fc"), ("gb", "fc"), ("mlp", "fl"), ("gb", "fl")]


def build():
    ab = S.parse_bracket(S.RUNS / "run_ablation_results.txt")
    body = []
    for i, (model, task) in enumerate(ORDER):
        if i == 2:
            body.append("\\midrule")
        a = S.ablation_agg(ab, task, model, ["20", "50"])
        # improvement magnitude (positive = improvement for both tasks), as in the condensed table
        body.append("%s & %s & %s & %s & +%s & %s \\\\" % (
            MODEL_AC[model], TASK_AC[task],
            S.fnum(a["default"], 3), S.fnum(a["best"], 3),
            S.fnum(a["dbest"], 3), S.fnum(a["spread"], 3)))

    latex = r"""\begin{table}[pos=ht]
\caption{Compact hyperparameter robustness summary for the representative \ac{mlp} and histogram-based \ac{gb} models based on the single-parameter ablations reported in this appendix. Ablation default refers to the best tagged default result within the independent ablation campaign across the reported decision horizons. Best denotes the strongest result observed in the corresponding ablation setting. For \ac{fc}, $\Delta_{\text{best}}$ denotes the increase in macro-\ac{f1}; for \ac{fl}, it denotes the reduction in \ac{mae}. Positive values therefore indicate improvement in both tasks.}
\small
\label{tab:ablation_robustness_summary_main}
\centering
\begin{tabular}{llrrrr}
\toprule
\textbf{Model} & \textbf{Task} & \textbf{Ablation default} & \textbf{Best} & \textbf{$\Delta_{\text{best}}$} & \textbf{Max. spread} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("ablation_robustness_summary_main", latex, "run_ablation_results.txt (committed)")


if __name__ == "__main__":
    build()
