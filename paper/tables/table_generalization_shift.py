"""tab:generalization_shift -- generalization across disjoint fault-resistance ranges.

In-distribution column = 20 ms episode-grouped reference (repro_fc/fl). Shifted columns come
from reports/runs/b4_generalization_seeds_results.txt: the \\ac{mlp} rows report the mean
$\\pm$ standard deviation (population, ddof=0) over five explicitly varied model seeds (0--4);
the histogram-based \\ac{gb} rows report the single deterministic seed-42 run. All committed.
"""
import sys
from pathlib import Path
from statistics import mean, pstdev
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FC_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier"}
FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor"}


def build():
    fc = S.parse_repro(S.RUNS / "repro_fc.txt")
    fl = S.parse_repro(S.RUNS / "repro_fl.txt")
    gen = S.parse_generalization(S.RUNS / "b4_generalization_seeds_results.txt")

    def indist(task, model):
        src, key = (fc, FC_KEY) if task == "fc" else (fl, FL_KEY)
        tgt = "event_type" if task == "fc" else "y_fault_location"
        return src[(tgt, key[model], "W=0.020")]["mean"]

    def shift_cell(task, model, direction, dp):
        vals = gen[(task, model, direction)]["vals"]
        if len(vals) > 1:  # MLP: 5 seeds -> mean +/- population std
            return "$%s\\pm%s$" % (S.fnum(mean(vals), dp), S.fnum(pstdev(vals), dp))
        return S.fnum(vals[0], dp)  # GB: single deterministic seed-42 run

    def row(task, model, tasklabel, dp):
        cells = [S.fnum(indist(task, model), dp),
                 shift_cell(task, model, "high", dp),
                 shift_cell(task, model, "low", dp)]
        return "%s & \\ac{%s} & %s \\\\" % (tasklabel, model, " & ".join(cells))

    body = [
        row("fc", "mlp", "\\ac{fc} (macro-\\ac{f1})", 3),
        row("fc", "gb",  "\\ac{fc} (macro-\\ac{f1})", 3),
        "\\midrule",
        row("fl", "mlp", "\\ac{fl} (\\ac{mae} [\\%])", 2),
        row("fl", "gb",  "\\ac{fl} (\\ac{mae} [\\%])", 2),
    ]

    latex = r"""\begin{table}[pos=ht]
\caption{Directional fault-resistance holdout at 20\,ms. Models are trained on the lower 80\%% and tested on the upper quintile of $R_f$, or vice versa. \ac{fc}: macro-\ac{f1}; \ac{fl}: \ac{mae} [\%% line length]. The episode-grouped five-fold result is shown for context and is not a training-size-matched control. Shifted \ac{mlp} columns are mean\,$\pm$\,standard deviation across five explicitly varied model seeds (0--4); the histogram-based \ac{gb} columns report one run with seed 42.}
\label{tab:generalization_shift}
\small
\centering
\begin{tabular}{llccc}
\toprule
\textbf{Task} & \textbf{Model} & \textbf{Five-fold reference} & \textbf{High-$R_f$ test} & \textbf{Low-$R_f$ test} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("generalization_shift", latex,
                  "repro (in-dist) + b4_generalization_seeds (MLP mean+/-std, GB seed 42) (committed)")


if __name__ == "__main__":
    build()
