"""tab:class_distribution -- class distribution of the 20 ms FC windows.

Source: reports/runs/class_split_stats.txt. Committed class ids are mapped to the
paper's class naming/order (AB/BC/CA, ABG/BCG/CAG); shares are recomputed from the
counts. All committed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sources as S

# (display label, type, [committed class ids in paper display order])
GROUPS = [
    ("No fault",        "--",      [0]),
    ("AG / BG / CG",    "1ph--g",  [1, 2, 3]),
    ("AB / BC / CA",    "2ph",     [4, 6, 5]),    # AB, BC, CA(=AC)
    ("ABG / BCG / CAG", "2ph--g",  [7, 9, 8]),    # ABG, BCG, CAG(=ACG)
    ("ABC",             "3ph",     [10]),
]


def num(n):
    return "{:,}".format(n).replace(",", "\\,") if n >= 10000 else str(n)


def build():
    cs = S.parse_class_stats(S.RUNS / "class_split_stats.txt")
    total = sum(v["n"] for v in cs.values())

    body = []
    for label, typ, ids in GROUPS:
        counts = " / ".join(num(cs[i]["n"]) for i in ids)
        shares = " / ".join(S.fnum(cs[i]["n"] / total * 100, 2) for i in ids)
        body.append("%-15s & %-8s & %-19s & %s\\%% \\\\" % (label, typ, counts, shares))
    body.append("\\midrule")
    body.append("%-15s & %-8s & %-19s & 100\\%% \\\\" % ("Total", "", num(total)))

    latex = r"""\begin{table}[ht]
\centering
\small
\caption{Class distribution of the 20\,ms fault-classification windows (\texttt{event\_type}; $N = %s$ windows over $9{,}022$ episodes).}
\label{tab:class_distribution}
\begin{tabular}{llrr}
\toprule
\textbf{Class} & \textbf{Type} & \textbf{Windows} & \textbf{Share} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}""" % ("{:,}".format(total).replace(",", "{,}"), "\n".join(body))

    S.write_table("class_distribution", latex, "class_split_stats.txt (committed)")


if __name__ == "__main__":
    build()
