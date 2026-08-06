"""tab:fl_observability_detail_mlp_50ms (Appendix A1) -- per-line 50 ms MLP-FL detail.

Source: reports/runs/run_obs_results.txt (obs_fl_mlp_{s,p}*_W50). Each protected line
maps to a relay pair (Terminal 1 = single relay a, Terminal 2 = single relay b,
Two-terminal = the same-line pair). Improvement = min(T1, T2) - two-terminal.
All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

# (line label, terminal-1 relay index, terminal-2 relay index)
LINES = [("01--02A", 0, 2), ("01--02B", 1, 3), ("02--03A", 4, 6), ("02--03B", 5, 7)]


def build():
    obs = S.parse_bracket(S.RUNS / "run_obs_results.txt")

    def single(i):
        return obs["obs_fl_mlp_s%d_W50" % i]["mean"]

    def pair(a, b):
        return obs["obs_fl_mlp_p%d-%d_W50" % (a, b)]["mean"]

    body = []
    for label, a, b in LINES:
        t1, t2, two = single(a), single(b), pair(a, b)
        improvement = min(t1, t2) - two
        body.append("%s & %s & %s & %s & %s \\\\" % (
            label, S.fnum(t1, 3), S.fnum(t2, 3), S.fnum(two, 3), S.fnum(improvement, 3)))

    latex = r"""\begin{table}[ht]
\centering
\small
\caption{Relay-level \ac{fl} error for the 50\,ms \ac{mlp} regressor under reduced observability. The improvement is computed relative to the better of the two single-terminal settings for each line.}
\label{tab:fl_observability_detail_mlp_50ms}
\begin{tabular}{lcccc}
\toprule
\textbf{Line} &
\multicolumn{2}{c}{\textbf{Single-terminal MAE [\%%]}} &
\textbf{Two-terminal} &
\textbf{Improvement} \\
\cmidrule(lr){2-3}
& \textbf{Terminal 1} & \textbf{Terminal 2} & \textbf{MAE [\%%]} & \textbf{[\%% points]} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % "\n".join(body)

    S.write_table("fl_observability_detail_mlp_50ms", latex,
                  "run_obs obs_fl_mlp_{s,p}*_W50 (committed)")


if __name__ == "__main__":
    build()
