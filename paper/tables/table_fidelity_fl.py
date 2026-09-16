"""tab:fidelity_fl -- FL robustness to measurement degradation at 20 ms (revblock).

Clean row = the five-fold reference statistic (repro_fl W20, mean $\\pm$ std). Perturbed levels =
reports/perturbation/<model>_y_fault_location_W20ms/perturbation_summary.csv (mean/std over
5 folds x 5 realizations). MLP CT c=0.3 (worst FL cell) bolded, matching the manuscript.
All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

REPRO_KEY = {"mlp": "mlp_regressor", "gb": "hist_gradient_boosting_regressor"}
# (axis label, level label, (pert_axis, pert_level) or None for clean, bold_mlp, bold_gb)
ROWS = [
    ("Clean",               "--",        None,                     False, False),
    ("Noise",               "20\\,dB",   ("noise", "20"),          False, False),
    ("Noise",               "10\\,dB",   ("noise", "10"),          False, False),
    ("\\ac{ct} saturation", "$c=0.5$",   ("ct_saturation", "0.5"), False, False),
    ("\\ac{ct} saturation", "$c=0.3$",   ("ct_saturation", "0.3"), True,  False),
    ("Jitter",              "2 samples", ("jitter", "2"),          False, False),
    ("Jitter",              "4 samples", ("jitter", "4"),          False, False),
]
DP = 2


def cell(v, bold):
    s = "%s $\\pm$ %s" % (S.fnum(v["mean"], DP), S.fnum(v["std"], DP))
    return "\\textbf{%s}" % s if bold else s


def build():
    repro = S.parse_repro(S.RUNS / "repro_fl.txt")
    mlp = S.load_perturbation("mlp_regressor_y_fault_location_W20ms")
    gb = S.load_perturbation("hist_gradient_boosting_regressor_y_fault_location_W20ms")
    clean = {"mlp": repro[("y_fault_location", REPRO_KEY["mlp"], "W=0.020")],
             "gb": repro[("y_fault_location", REPRO_KEY["gb"], "W=0.020")]}

    body = []
    for axis, level, key, bm, bg in ROWS:
        mv, gv = (clean["mlp"], clean["gb"]) if key is None else (mlp[key], gb[key])
        body.append("%-21s & %-9s & %s & %s \\\\" % (axis, level, cell(mv, bm), cell(gv, bg)))

    latex = r"""\begin{table}[pos=ht]
\caption{\ac{fl} robustness to measurement degradation at 20\,ms. The clean row reproduces the five-fold reference statistic. For each perturbed level, the five realizations are first averaged within each fold; values are then reported as mean $\pm$ standard deviation across the five fold-level means. Lower is better.}
\label{tab:fidelity_fl}
\small
\centering
\begin{revblock}
\begin{tabular}{llcc}
\toprule
\textbf{Axis} & \textbf{Level} & \textbf{\ac{mlp}} & \textbf{\ac{gb}} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{revblock}
\end{table}""" % "\n".join(body)

    S.write_table("fidelity_fl", latex,
                  "repro_fl clean + perturbation_summary.csv FL MLP+GB W20 (committed)")


if __name__ == "__main__":
    build()
