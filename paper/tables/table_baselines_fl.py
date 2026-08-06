"""tab:baselines_fl -- FL learning vs conventional impedance methods (revblock).

Learning rows: repro_fl (W=0.020 / 0.050). Conventional rows: reports/baselines/
fl_{two,one}_ended_W{20,50}ms/cv_summary.json (per-window `mae` + `mae_settled_per_episode`).
Two-ended settled is bolded. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor",
          "knn": "k_neighbors_regressor", "ridge": "ridge_regressor"}


def mae(m, s, dp, phantom=False, bold=False):
    inner = "%s\\pm%s" % (S.fnum(m, dp), S.fnum(s, dp))
    if bold:
        inner = "\\mathbf{%s}" % inner
    return "$%s%s$" % ("\\phantom{0}" if phantom else "", inner)


def learn(v20, v50):  # 2 dp, phantom on single-digit means
    return (mae(v20["mean"], v20["std"], 2, phantom=v20["mean"] < 10),
            mae(v50["mean"], v50["std"], 2, phantom=v50["mean"] < 10))


def build():
    fl = S.parse_repro(S.RUNS / "repro_fl.txt")
    two20, two50 = S.load_baseline("fl_two_ended_W20ms"), S.load_baseline("fl_two_ended_W50ms")
    one20, one50 = S.load_baseline("fl_one_ended_W20ms"), S.load_baseline("fl_one_ended_W50ms")

    def lrow(name, key):
        c20, c50 = learn(fl[("y_fault_location", key, "W=0.020")],
                          fl[("y_fault_location", key, "W=0.050")])
        return "%s        & full        & %s & %s \\\\" % (name, c20, c50)

    body = [
        lrow("\\ac{mlp}", FL_KEY["mlp"]),
        lrow("\\ac{gb}", FL_KEY["gb"]),
        lrow("\\ac{knn}", FL_KEY["knn"]),
        lrow("Ridge", FL_KEY["ridge"]),
        "\\midrule",
        "Two-ended (per window) & relay pair   & %s & %s \\\\" % (
            mae(two20["mae"]["mean"], two20["mae"]["std"], 2, phantom=True),
            mae(two50["mae"]["mean"], two50["mae"]["std"], 2, phantom=True)),
        "Two-ended (settled)    & relay pair   & %s & %s \\\\" % (
            mae(two20["mae_settled_per_episode"]["mean"], two20["mae_settled_per_episode"]["std"], 2, phantom=True, bold=True),
            mae(two50["mae_settled_per_episode"]["mean"], two50["mae_settled_per_episode"]["std"], 2, phantom=True, bold=True)),
        "One-ended (per window) & single relay & %s & %s \\\\" % (
            mae(one20["mae"]["mean"], one20["mae"]["std"], 1),
            mae(one50["mae"]["mean"], one50["mae"]["std"], 1)),
        "One-ended (settled)    & single relay & %s & %s \\\\" % (
            mae(one20["mae_settled_per_episode"]["mean"], one20["mae_settled_per_episode"]["std"], 1),
            mae(one50["mae_settled_per_episode"]["mean"], one50["mae_settled_per_episode"]["std"], 1)),
    ]

    latex = r"""\begin{table}[ht]
\caption{\rev{\ac{fl} performance of learning and conventional impedance-based methods under shared splits, validity rules, and metrics. \ac{mae} [\%% line length], mean $\pm$ standard deviation over five folds. Conventional locators additionally use per-episode line parameters and ground-truth loop selection; results are reported per window and per episode using the settled onset window.}}
\label{tab:baselines_fl}
\small
\centering
\begin{revblock}
\begin{tabular}{llcc}
\toprule
\textbf{Method} & \textbf{Observability} & \textbf{20\,ms} & \textbf{50\,ms} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{revblock}
\end{table}""" % "\n".join(body)

    S.write_table("baselines_fl", latex, "repro_fl + baselines/*_ended cv_summary.json (committed)")


if __name__ == "__main__":
    build()
