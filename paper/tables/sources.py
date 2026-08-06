"""Shared source loaders + LaTeX formatters for the paper table generators.

Every number in the generated `table__*.tex` files is extracted here from the
committed evidence under ``reports/`` -- nothing is hand-typed except the
clearly-labelled cells in ``legacy_values.py`` (cells with no committed source).

Parsers mirror the committed log formats exactly:
  * ``reports/runs/*.txt``            -> bracket-tag lines ``[tag] macroF1|MAE mean=.. std=..``
  * ``reports/runs/repro_*.txt``      -> ``=== <target> <model> W=..: <metric> mean=.. std=..``
  * ``reports/runs/run_runtime_*``    -> ``[tag] train_s=.. infer_us=.. thr_ks=.. node=..``
  * ``reports/runs/b4_generalization*``-> ``[tag] <target> <model> W=.. shift=.. [seed=..] .. metric=..``
  * ``reports/baselines/*/cv_summary.json``
  * ``reports/perturbation/*/perturbation_summary.csv``

Std is population std (ddof=0), matching the harnesses (numpy default) -- verified
against the committed per-fold values.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
RUNS = REPORTS / "runs"
OUT_DIR = Path(__file__).resolve().parent

# Collected across a build so build_all.py can print one provenance summary.
WARNINGS: list[str] = []


# --------------------------------------------------------------------------- #
# statistics (ddof=0)
# --------------------------------------------------------------------------- #
def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def pstd(xs) -> float:
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
_BRACKET = re.compile(
    r"^\[(?P<tag>[^\]]+)\]\s+(?P<metric>macroF1|MAE)\s+"
    r"mean=(?P<mean>[-+0-9.eE]+)\s+std=(?P<std>[-+0-9.eE]+)"
)

_REPRO = re.compile(
    r"^===\s+(?P<target>\S+)\s+(?P<model>\S+)\s+W=(?P<w>[0-9.]+):\s+"
    r"(?P<metric>macroF1|MAE)\s+mean=(?P<mean>[-+0-9.eE]+)\s+std=(?P<std>[-+0-9.eE]+)"
)

_RUNTIME = re.compile(
    r"^\[(?P<tag>[^\]]+)\]\s+train_s=(?P<train>[-+0-9.eE]+)\s+"
    r"infer_us=(?P<infer>[-+0-9.eE]+)\s+thr_ks=(?P<thr>[-+0-9.eE]+).*?"
    r"node=(?P<node>\S+)\s+cpus=(?P<cpus>\d+)\s+omp=(?P<omp>\d+)"
)

_GEN = re.compile(
    r"^\[(?P<tag>[^\]]+)\]\s+(?P<target>\S+)\s+(?P<model>\S+)\s+W=(?P<w>[0-9.]+)\s+"
    r"shift=(?P<shift>\w+)(?:\s+seed=(?P<seed>\d+))?\s+thr=[-+0-9.eE]+\s+"
    r"n_train=(?P<ntr>\d+)\s+n_test=(?P<nte>\d+)\s+(?P<metric>macroF1|MAE)=(?P<val>[-+0-9.eE]+)"
)


def parse_bracket(path) -> dict:
    """``[tag] macroF1|MAE mean=.. std=..`` -> {tag: {metric, mean, std}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _BRACKET.match(line.strip())
        if m:
            out[m["tag"]] = {"metric": m["metric"], "mean": float(m["mean"]), "std": float(m["std"])}
    return out


def parse_repro(path) -> dict:
    """``=== <target> <model> W=..: ..`` -> {(target, model, 'W=0.020'): {mean, std}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _REPRO.match(line.strip())
        if m:
            out[(m["target"], m["model"], f"W={m['w']}")] = {
                "mean": float(m["mean"]), "std": float(m["std"])}
    return out


def parse_runtime(path) -> dict:
    """runtime log -> {tag: {train, infer, thr, node, cpus, omp}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _RUNTIME.match(line.strip())
        if m:
            out[m["tag"]] = {
                "train": float(m["train"]), "infer": float(m["infer"]), "thr": float(m["thr"]),
                "node": m["node"], "cpus": int(m["cpus"]), "omp": int(m["omp"])}
    return out


def parse_generalization(path) -> dict:
    """b4 log -> {(task, model, shift): {'vals': [..], 'n_test': int}}.

    task in {fc, fl}; model in {mlp, gb}; shift in {high, low}. MLP has 5 seeds,
    GB a single deterministic run.
    """
    tmap = {"event_type": "fc", "y_fault_location": "fl"}
    mmap = {"mlp_classifier": "mlp", "mlp_regressor": "mlp",
            "hist_gradient_boosting_classifier": "gb", "hist_gradient_boosting_regressor": "gb"}
    out: dict = {}
    for line in Path(path).read_text().splitlines():
        m = _GEN.match(line.strip())
        if not m:
            continue
        key = (tmap[m["target"]], mmap[m["model"]], m["shift"])
        rec = out.setdefault(key, {"vals": [], "n_test": int(m["nte"])})
        rec["vals"].append(float(m["val"]))
    return out


def load_baseline(name: str) -> dict:
    """``reports/baselines/<name>/cv_summary.json`` (e.g. ``fl_two_ended_W20ms``)."""
    return json.loads((REPORTS / "baselines" / name / "cv_summary.json").read_text())


def load_perturbation(name: str) -> dict:
    """``reports/perturbation/<name>/perturbation_summary.csv`` -> {(axis, level): {mean, std, count}}."""
    rows = {}
    with open(REPORTS / "perturbation" / name / "perturbation_summary.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["axis"], r["level"])] = {
                "mean": float(r["mean"]), "std": float(r["std"]), "count": int(r["count"])}
    return rows


# CNN / MOMENT lines carry the metric mid-line (after tag + target + model + mode),
# e.g. "[moment_fc_W50_probe] event_type moment W=0.050 mode=probe macroF1 mean=.. std=..".
_LABELED = re.compile(
    r"^\[(?P<tag>[^\]]+)\].*?\b(?P<metric>macroF1|MAE)\s+"
    r"mean=(?P<mean>[-+0-9.eE]+)\s+std=(?P<std>[-+0-9.eE]+)"
)


# --- Hyperparameter-ablation aggregation (mirrors docs/claims.py _b_ablation) ---
ABL_GROUPS = {
    "gb": [("learn. rate", ["default", "lr003", "lr02"]),
           ("depth",       ["default", "d3", "d5", "d10"]),
           ("iters",       ["default", "it50", "it300"]),
           ("min leaf",    ["default", "ml5", "ml50"]),
           ("L2 reg.",     ["default", "l2_1e4", "l2_1e2"])],
    "mlp": [("hidden",     ["default", "hid50", "hid100_50", "hid256_128"]),
            ("L2 penalty", ["default", "alpha1e5", "alpha1e3"]),
            ("init. lr",   ["default", "ilr1e5", "ilr1e4", "ilr1e2"]),
            ("batch",      ["default", "bs64", "bs128", "bs256"]),
            ("iters",      ["default", "it100", "it300", "it400"])],
}


def _abl_grp(model, suffix):
    for glabel, sfxs in ABL_GROUPS[model]:
        if suffix in sfxs and suffix != "default":
            return glabel
    return suffix


def ablation_agg(ab, task, model, windows):
    """Aggregate one ablation cell (task, model, over window(s)) from parse_bracket(run_ablation_*).
    Returns {default, best, dbest, dbest_knob, spread, spread_knob}. default = best default among windows;
    dbest = best-default (FC) / default-best (FL); spread = largest best-worst within one knob group."""
    cells = {}
    for w in windows:
        pfx, sfx = "abl_%s_%s_" % (task, model), "_W%s" % w
        for tag, node in ab.items():
            if tag.startswith(pfx) and tag.endswith(sfx):
                cells[(w, tag[len(pfx):-len(sfx)])] = node["mean"]
    f = max if task == "fc" else min
    defs = {k: v for k, v in cells.items() if k[1] == "default"}
    dkey = f(defs, key=defs.get)
    bkey = f(cells, key=cells.get)
    dval, bval = defs[dkey], cells[bkey]
    best_spread, sp_label = -1.0, ""
    for w in windows:
        wv = {s: cells[(w, s)] for (ww, s) in cells if ww == w}
        for glabel, sfxs in ABL_GROUPS[model]:
            g = [wv[s] for s in sfxs if s in wv]
            if len(g) >= 2 and max(g) - min(g) > best_spread:
                best_spread, sp_label = max(g) - min(g), glabel
    return {"default": dval, "best": bval,
            "dbest": (bval - dval) if task == "fc" else (dval - bval),
            "dbest_knob": _abl_grp(model, bkey[1]), "spread": best_spread, "spread_knob": sp_label}


def parse_labeled(path) -> dict:
    """CNN/MOMENT-style tagged lines -> {tag: {metric, mean, std}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _LABELED.match(line.strip())
        if m:
            out[m["tag"]] = {"metric": m["metric"], "mean": float(m["mean"]), "std": float(m["std"])}
    return out


_CLASS = re.compile(r"^\s*(?P<name>\S+)\s+id=\s*(?P<id>\d+)\s+n=\s*(?P<n>\d+)\s+(?P<pct>[\d.]+)%")


def parse_class_stats(path) -> dict:
    """reports/runs/class_split_stats.txt per-class lines -> {id: {name, n, pct}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _CLASS.match(line)
        if m:
            out[int(m["id"])] = {"name": m["name"], "n": int(m["n"]), "pct": float(m["pct"])}
    return out


# --------------------------------------------------------------------------- #
# observability aggregation (mean over the reduced-sensing runs)
# --------------------------------------------------------------------------- #
def obs_group_mean(obs: dict, task: str, model: str, w: str, kind: str, expect: int) -> float:
    """Mean of the per-run means for a reduced-observability group.

    kind='s' -> the 8 single-relay runs; kind='p' -> the 4 same-line relay pairs.
    ``w`` is '20' or '50'. Raises if the group size is unexpected (guards a silent
    partial aggregate).
    """
    prefix = f"obs_{task}_{model}_{kind}"
    suffix = f"_W{w}"
    vals = [v["mean"] for tag, v in obs.items() if tag.startswith(prefix) and tag.endswith(suffix)]
    if len(vals) != expect:
        raise ValueError(f"obs group {prefix}*{suffix}: expected {expect} runs, found {len(vals)}")
    return mean(vals)


def obs_group_stats(obs: dict, task: str, model: str, w: str, kind: str, expect: int):
    """(mean, lo, hi) over the per-config means for a reduced-observability group.

    Same grouping as ``obs_group_mean``; additionally returns the min and max of the
    per-config (per-relay / per-pair) means, i.e. the min--max range that quantifies
    relay-location spread for the aggregated row.
    """
    prefix = f"obs_{task}_{model}_{kind}"
    suffix = f"_W{w}"
    vals = [v["mean"] for tag, v in obs.items() if tag.startswith(prefix) and tag.endswith(suffix)]
    if len(vals) != expect:
        raise ValueError(f"obs group {prefix}*{suffix}: expected {expect} runs, found {len(vals)}")
    return mean(vals), min(vals), max(vals)


# --------------------------------------------------------------------------- #
# LaTeX formatting
# --------------------------------------------------------------------------- #
def fnum(x: float, dp: int) -> str:
    return f"{x:.{dp}f}"


def pm(mean_: float, std_: float, dp: int) -> str:
    return f"{fnum(mean_, dp)} $\\pm$ {fnum(std_, dp)}"


def signed(x: float, dp: int) -> str:
    """Signed delta, e.g. +0.221 / -0.007 / +0.00."""
    s = f"{x:+.{dp}f}"
    if float(s) == 0:  # normalise -0.000 -> +0.000
        s = f"{0.0:+.{dp}f}"
    return s


def warn(msg: str) -> None:
    WARNINGS.append(msg)


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
def write_table(name: str, latex: str, provenance: str) -> Path:
    path = OUT_DIR / f"table__{name}.tex"
    path.write_text(latex.rstrip() + "\n")
    print(f"  wrote table__{name}.tex  --  {provenance}")
    return path
