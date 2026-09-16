"""tab:fc_observability -- macro-F1 under full / relay-pair / single-relay sensing.

Full = the 20/50 ms reference (repro_fc), reported as mean $\\pm$ standard deviation over
the five episode-grouped folds. Relay pair / single relay = mean over the 4 same-line pairs /
8 single-relay runs (run_obs_results.txt), reported as the mean with the min--max range across
those configurations in brackets (a fold std would understate the relay-location spread).
Column-best (max mean) is bolded. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

FC_KEY = {"mlp": "mlp_classifier", "gb": "hist_gradient_boosting_classifier"}
COLS = [("mlp", "20"), ("gb", "20"), ("mlp", "50"), ("gb", "50")]
DP = 3


def build():
    repro = S.parse_repro(S.RUNS / "repro_fc.txt")
    obs = S.parse_bracket(S.RUNS / "run_obs_results.txt")

    full = {c: repro[("event_type", FC_KEY[c[0]], "W=0.0%s" % c[1])] for c in COLS}
    pair = {c: S.obs_group_stats(obs, "fc", c[0], c[1], "p", 4) for c in COLS}
    single = {c: S.obs_group_stats(obs, "fc", c[0], c[1], "s", 8) for c in COLS}

    means = {"Full": {c: full[c]["mean"] for c in COLS},
             "Relay pair": {c: pair[c][0] for c in COLS},
             "Single relay": {c: single[c][0] for c in COLS}}
    best = {c: max(means, key=lambda r: means[r][c]) for c in COLS}  # max macro-F1

    def full_cell(c):
        s = "%s $\\pm$ %s" % (S.fnum(full[c]["mean"], DP), S.fnum(full[c]["std"], DP))
        return "\\textbf{%s}" % s if best[c] == "Full" else s

    def range_cell(c, d, row):
        mn, lo, hi = d[c]
        m = "\\textbf{%s}" % S.fnum(mn, DP) if best[c] == row else S.fnum(mn, DP)
        return "%s (%s--%s)" % (m, S.fnum(lo, DP), S.fnum(hi, DP))

    body = [
        "Full         & %s \\\\" % " & ".join(full_cell(c) for c in COLS),
        "Relay pair   & %s \\\\" % " & ".join(range_cell(c, pair, "Relay pair") for c in COLS),
        "Single relay & %s \\\\" % " & ".join(range_cell(c, single, "Single relay") for c in COLS),
    ]

    latex = r"""\begin{table*}[pos=t]
\caption{Controlled observability sensitivity for fault classification. Macro-\ac{f1} under full, relay-pair, and single-relay sensing for \ac{mlp} and histogram-based \ac{gb} models at 20\,ms and 50\,ms. Full-observability values are the mean $\pm$ standard deviation over the five episode-grouped folds; the relay-pair and single-relay values are the mean over the four same-line relay pairs and the eight individual relays, respectively, with the min--max range across those configurations in brackets. Higher is better.}
\small
\label{tab:fc_observability}
\centering
\begin{tabular}{lcccc}
\toprule
\bfseries Observability & \bfseries \ac{mlp} (20\,ms) & \bfseries \ac{gb} (20\,ms) & \bfseries \ac{mlp} (50\,ms) & \bfseries \ac{gb} (50\,ms) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}""" % "\n".join(body)

    S.write_table("fc_observability", latex,
                  "run_obs aggregate (mean + min--max range) + repro_fc full (mean +/- std) (committed)")


if __name__ == "__main__":
    build()
