"""Aggregate the 20 per-fold MOMENT JSON parts into the parser-compatible 8-line result file.

Each corrected MOMENT job writes one JSON (``<tag>_fold<k>.json``) for one outer fold. This
script reads exactly 20 of them (4 configurations x 5 folds), validates fold completeness and the
leakage guards recorded in every part, then emits the eight aggregate lines consumed by
``docs/gen_site_data.py`` (MOMENT_RE) and ``paper/tables/sources.py`` (_LABELED).

    python runs/harnesses/aggregate_moment_results.py \
        --parts /path/to/parts \
        --out reports/runs/moment_results_candidate.txt \
        --strict
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

EXPECTED_TAGS = ("moment_fc_W20", "moment_fc_W50", "moment_fl_W20", "moment_fl_W50")
EXPECTED_FOLDS = [1, 2, 3, 4, 5]
FINAL_TAG_ORDER = [f"{t}_{mode}" for t in EXPECTED_TAGS for mode in ("probe", "head")]


class AggregationError(Exception):
    """Raised when the parts fail a completeness or leakage-guard check under strict mode."""


def load_parts(parts_dir):
    """Load every ``*.json`` part in ``parts_dir`` (non-recursive). Returns a list of dicts."""
    paths = sorted(glob.glob(os.path.join(parts_dir, "*.json")))
    records = []
    for path in paths:
        with open(path) as f:
            rec = json.load(f)
        rec["_path"] = path
        records.append(rec)
    return records


def _fail(strict, errors, msg):
    errors.append(msg)
    if strict:
        raise AggregationError(msg)


def _metric_name(task):
    return "macroF1" if task == "clf" else "MAE"


def _fmt_line(tag, mode, target, wl, metric, scores):
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=0))
    per_fold = [round(float(x), 4) for x in scores]
    return (f"[{tag}_{mode}] {target} moment W={wl:.3f} mode={mode} {metric} "
            f"mean={mean:.4f} std={std:.4f} per_fold={per_fold} device=cuda")


def aggregate(records, strict=True):
    """Validate the parts and return the eight aggregate lines (in FINAL_TAG_ORDER).

    Under ``strict`` any failed check raises ``AggregationError``; otherwise checks are collected
    and printed as warnings and aggregation proceeds on whatever is present.
    """
    errors = []

    by_tag = {}
    for rec in records:
        by_tag.setdefault(rec["tag"], []).append(rec)

    got_tags = set(by_tag)
    expected = set(EXPECTED_TAGS)
    if got_tags != expected:
        _fail(strict, errors,
              f"config set mismatch: missing={sorted(expected - got_tags)} extra={sorted(got_tags - expected)}")

    lines = []
    for tag in EXPECTED_TAGS:
        recs = by_tag.get(tag, [])
        folds = sorted(r["outer_fold"] for r in recs)
        if folds != EXPECTED_FOLDS:
            dups = sorted({f for f in folds if folds.count(f) > 1})
            missing = sorted(set(EXPECTED_FOLDS) - set(folds))
            _fail(strict, errors,
                  f"[{tag}] folds != 1..5 (got {folds}; missing={missing} duplicate={dups})")
            continue

        recs = sorted(recs, key=lambda r: r["outer_fold"])
        # Per-fold leakage guards.
        for r in recs:
            if r.get("group_overlap", -1) != 0:
                _fail(strict, errors, f"[{tag}] fold {r['outer_fold']} group_overlap={r.get('group_overlap')} != 0")
            if r.get("raw_scaler_fit_windows") != r.get("n_train_windows"):
                _fail(strict, errors,
                      f"[{tag}] fold {r['outer_fold']} raw_scaler_fit_windows="
                      f"{r.get('raw_scaler_fit_windows')} != n_train_windows={r.get('n_train_windows')}")
            for k in ("probe_score", "head_score"):
                if not np.isfinite(r.get(k, np.nan)):
                    _fail(strict, errors, f"[{tag}] fold {r['outer_fold']} {k} not finite ({r.get(k)})")
            if r.get("smoke"):
                _fail(strict, errors, f"[{tag}] fold {r['outer_fold']} is a SMOKE part (not a real result)")

        target = recs[0]["target"]
        task = recs[0]["task"]
        wl = float(recs[0]["window_length"])
        metric = _metric_name(task)
        probe_scores = [r["probe_score"] for r in recs]
        head_scores = [r["head_score"] for r in recs]
        lines.append(_fmt_line(tag, "probe", target, wl, metric, probe_scores))
        lines.append(_fmt_line(tag, "head", target, wl, metric, head_scores))

    if errors and not strict:
        for e in errors:
            print(f"WARNING: {e}", file=sys.stderr)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="directory of per-fold JSON parts")
    ap.add_argument("--out", default="reports/runs/moment_results_candidate.txt")
    ap.add_argument("--strict", action="store_true", help="hard-fail on any completeness/leakage check")
    args = ap.parse_args()

    records = load_parts(args.parts)
    n = len(records)
    if n != 20:
        msg = f"expected exactly 20 parts in {args.parts}, found {n}"
        if args.strict:
            raise SystemExit(f"ERROR: {msg}")
        print(f"WARNING: {msg}", file=sys.stderr)

    try:
        lines = aggregate(records, strict=args.strict)
    except AggregationError as exc:
        raise SystemExit(f"ERROR: {exc}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} aggregate lines -> {args.out}")
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
