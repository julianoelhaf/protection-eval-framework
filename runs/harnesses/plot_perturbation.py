"""Generate Experiment B figures from reports/perturbation/*/perturbation_summary.csv.

Fig B1/B2: metric vs SNR (clean anchor leftmost), MLP + GB overlaid, +/-1 std ribbon.
Also CT-saturation and jitter line plots. Saves PNGs to reports/perturbation/figures/.
Safe to run repeatedly; skips axes with no data.
"""
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = "/path/to/repos/protection-eval-framework"
OUT = f"{REPO}/reports/perturbation/figures"
os.makedirs(OUT, exist_ok=True)

SNR_ORDER = {"clean": 0, "40": 1, "30": 2, "20": 3, "10": 4}
CT_ORDER = {"no_sat": 0, "0.7": 1, "0.5": 2, "0.3": 3}
JIT_ORDER = {"0": 0, "1": 1, "2": 2, "4": 3}


def load():
    runs = {}
    for p in glob.glob(f"{REPO}/reports/perturbation/*/perturbation_summary.csv"):
        name = os.path.basename(os.path.dirname(p))
        m = re.match(r"(.+?)_(y_fault_location|event_type)_W(\d+)ms", name)
        if not m:
            continue
        model, target, ms = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(p)
        runs[(model, target, ms)] = df
    return runs


def _order(axis):
    return {"noise": SNR_ORDER, "ct_saturation": CT_ORDER, "jitter": JIT_ORDER}[axis]


def plot_axis(runs, axis, target, ms, xlabel, ylabel, fname, title):
    order = _order(axis)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for (model, tgt, mms), df in sorted(runs.items()):
        if tgt != target or mms != ms:
            continue
        sub = df[df["axis"] == axis].copy()
        if sub.empty:
            continue
        sub["ord"] = sub["level"].astype(str).map(order)
        sub = sub.dropna(subset=["ord"]).sort_values("ord")
        x = range(len(sub))
        ax.plot(x, sub["mean"], marker="o", label=model.replace("_regressor", "").replace("_classifier", ""))
        ax.fill_between(x, sub["mean"] - sub["std"].fillna(0), sub["mean"] + sub["std"].fillna(0), alpha=0.2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["level"].astype(str))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"{OUT}/{fname}"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    runs = load()
    if not runs:
        print("no perturbation summaries yet")
        return
    made = []
    for target, metric_lbl, tag in [("y_fault_location", "MAE [% line]", "FL"),
                                     ("event_type", "Macro-F1", "FC")]:
        for ms in (20, 50):
            for axis, xl, fn in [("noise", "SNR [dB]", f"{tag}_noise_W{ms}ms.png"),
                                 ("ct_saturation", "CT retained fraction c", f"{tag}_ct_W{ms}ms.png"),
                                 ("jitter", "jitter delta_max [samples]", f"{tag}_jitter_W{ms}ms.png")]:
                p = plot_axis(runs, axis, target, ms, xl, metric_lbl, fn, f"{tag} {axis} ({ms} ms)")
                if p:
                    made.append(p)
    print(f"wrote {len(made)} figures to {OUT}")
    for p in made:
        print(" ", os.path.basename(p))


if __name__ == "__main__":
    main()
