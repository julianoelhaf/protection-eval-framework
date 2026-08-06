"""tab:baselines_obs -- FL by observability at 20 ms, sensing-matched (revblock).

Conv. column: two-ended settled (relay pair) + one-ended settled (single relay) from
the baseline cv_summary.json. MLP/GB columns: the 20 ms full reference (repro_fl) and
the relay-pair / single-relay aggregates from run_obs_results.txt. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FL_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor"}


def build():
    fl = S.parse_repro(S.RUNS / "repro_fl.txt")
    obs = S.parse_bracket(S.RUNS / "run_obs_results.txt")
    two20 = S.load_baseline("fl_two_ended_W20ms")
    one20 = S.load_baseline("fl_one_ended_W20ms")

    mlp_full = fl[("y_fault_location", FL_KEY["mlp"], "W=0.020")]["mean"]
    gb_full = fl[("y_fault_location", FL_KEY["gb"], "W=0.020")]["mean"]
    conv_pair = two20["mae_settled_per_episode"]["mean"]     # 2.74
    conv_single = one20["mae_settled_per_episode"]["mean"]   # 41.9
    mlp_pair = S.obs_group_mean(obs, "fl", "mlp", "20", "p", 4)
    gb_pair = S.obs_group_mean(obs, "fl", "gb", "20", "p", 4)
    mlp_single = S.obs_group_mean(obs, "fl", "mlp", "20", "s", 8)
    gb_single = S.obs_group_mean(obs, "fl", "gb", "20", "s", 8)

    body = [
        "Full         & (n/a)                & --   & %s & %s \\\\" % (S.fnum(mlp_full, 2), S.fnum(gb_full, 2)),
        "Relay pair   & two-ended synchr.    & \\textbf{%s} & %s & %s \\\\" % (
            S.fnum(conv_pair, 2), S.fnum(mlp_pair, 2), S.fnum(gb_pair, 2)),
        "Single relay & one-ended reactance  & %s & %s & %s \\\\" % (
            S.fnum(conv_single, 1), S.fnum(mlp_single, 2), S.fnum(gb_single, 2)),
    ]

    latex = r"""\begin{table}[ht]
\caption{\rev{\ac{fl} performance by observability at 20\,ms for sensing-matched conventional and learning methods. \ac{mae} [\%% line length]; lower is better.}}
\label{tab:baselines_obs}
\small
\centering
\begin{revblock}
\begin{tabular}{llccc}
\toprule
\textbf{Observability} & \textbf{Conventional} & \textbf{Conv.} & \textbf{\ac{mlp}} & \textbf{\ac{gb}} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{revblock}
\end{table}""" % "\n".join(body)

    S.write_table("baselines_obs", latex, "repro_fl + run_obs aggregate + baselines settled (committed)")


if __name__ == "__main__":
    build()
