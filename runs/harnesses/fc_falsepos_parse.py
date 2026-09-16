"""Compute the no-fault false-positive / nuisance-trip rate for the FC task on
PROTECT-90 from a persisted OOF prediction dump.

Reads ``oof_predictions.parquet`` (columns: y_true, y_pred, status, fold, ...)
produced by ``fcl_psp.models.run_model`` and reports, per GroupKFold fold and
pooled across folds:

  naive FP rate    = mean(y_pred != 0 | y_true == 0)
      All windows the onset-only labeling collapses to class 0 (=no_fault).
      This MIXES genuine pre-fault nuisance trips with penalizing correct
      "fault" calls on in-fault windows, so it is an UPPER bound / diagnostic.

  honest nuisance  = mean(y_pred != 0 | status == 'clean')      <- headline
      Restricted to genuine PRE-FAULT ('clean') windows: the model sees no
      fault yet and any non-zero prediction is a true nuisance trip.

CPU-only, no training/inference; intended to run on a SLURM compute node (or via
a short srun). Depends only on pandas + numpy.

Usage:
    python fc_falsepos_parse.py --parquet <oof.parquet> --out <results.txt>
        [--jobid J] [--commit SHA] [--cmd "..."] [--no-fault-id 0]
"""
import argparse
import datetime as _dt

import numpy as np
import pandas as pd


def pct(x: float) -> str:
    return "nan" if x != x else f"{100.0 * x:.4f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="path to oof_predictions.parquet")
    ap.add_argument("--out", required=True, help="results text file to (over)write")
    ap.add_argument("--jobid", default="NA")
    ap.add_argument("--commit", default="NA")
    ap.add_argument("--cmd", default="NA")
    ap.add_argument("--clean-status", default="clean")
    ap.add_argument("--no-fault-id", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    need = {"y_true", "y_pred", "status", "fold"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(
            f"OOF parquet missing required columns {sorted(missing)}; "
            f"present: {list(df.columns)}"
        )

    yt = df["y_true"].to_numpy().astype(int)
    yp = df["y_pred"].to_numpy().astype(int)
    status = df["status"].astype(str).str.lower().to_numpy()
    fold = df["fold"].to_numpy().astype(int)
    nf = args.no_fault_id
    clean = args.clean_status.lower()

    folds = sorted(int(f) for f in np.unique(fold))
    status_vc = pd.Series(status).value_counts()

    def rate(num_mask, den_mask):
        den = int(den_mask.sum())
        num = int((num_mask & den_mask).sum())
        return num, den, (num / den if den else float("nan"))

    is_pred_fault = yp != nf
    is_no_fault = yt == nf
    is_clean = status == clean

    naive_pf, honest_pf = [], []
    per_fold_lines = []
    for f in folds:
        fm = fold == f
        n_num, n_den, n_r = rate(is_pred_fault, fm & is_no_fault)
        h_num, h_den, h_r = rate(is_pred_fault, fm & is_clean)
        naive_pf.append(n_r)
        honest_pf.append(h_r)
        per_fold_lines.append(
            f"  fold {f}:  naive_FP={pct(n_r):>10} ({n_num:>5}/{n_den:>6})   "
            f"nuisance_clean={pct(h_r):>10} ({h_num:>5}/{h_den:>6})"
        )

    naive_pf = np.asarray(naive_pf, dtype=float)
    honest_pf = np.asarray(honest_pf, dtype=float)

    n_num, n_den, n_r = rate(is_pred_fault, is_no_fault)  # pooled naive
    h_num, h_den, h_r = rate(is_pred_fault, is_clean)      # pooled honest

    def mean_std(a):
        s1 = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        s0 = float(np.std(a, ddof=0))
        return float(np.mean(a)), s1, s0

    n_mean, n_s1, n_s0 = mean_std(naive_pf)
    h_mean, h_s1, h_s0 = mean_std(honest_pf)

    W = str(0.020)
    out = []
    out.append(
        "FC no-fault false-positive / nuisance-trip rate "
        "(event_type, mlp_classifier, 20ms) - PROTECT-90"
    )
    out.append(f"  generated : {_dt.datetime.now().isoformat(timespec='seconds')}")
    out.append(f"  slurm_job : {args.jobid}")
    out.append(f"  git_commit: {args.commit}")
    out.append(f"  command   : {args.cmd}")
    out.append(f"  oof_dump  : {args.parquet}")
    out.append(f"  N windows : {len(df)}   folds: {folds}   no_fault_id: {nf}")
    out.append("")
    out.append("  Distinct window statuses present (status : count):")
    for name, cnt in status_vc.items():
        out.append(f"    {str(name):<14} n={int(cnt):>7}")
    out.append(f"    -> 'clean' (pre-fault) windows = {int(status_vc.get(clean, 0))}")
    out.append("")
    out.append("  Definitions:")
    out.append(
        "    naive FP rate   = mean(y_pred != 0 | y_true == 0)      "
        "[all no_fault-labeled windows; onset-only labeling folds pre-fault + in-fault together]"
    )
    out.append(
        "    honest nuisance = mean(y_pred != 0 | status == 'clean') "
        "[genuine PRE-FAULT windows only]  <-- headline / defensible metric"
    )
    out.append("")
    out.append("  Per-fold (5-fold GroupKFold):")
    out.extend(per_fold_lines)
    out.append("")
    out.append(f"  Across-fold summary (N={len(folds)} folds):")
    out.append(
        f"    naive FP rate   : mean={pct(n_mean)}  std={pct(n_s1)} (ddof=1) "
        f"[ddof=0 std={pct(n_s0)}]"
    )
    out.append(
        f"    honest nuisance : mean={pct(h_mean)}  std={pct(h_s1)} (ddof=1) "
        f"[ddof=0 std={pct(h_s0)}]"
    )
    out.append("")
    out.append("  Pooled across all 5 folds (single numerator/denominator):")
    out.append(f"    naive FP rate   : {pct(n_r)}  ({n_num}/{n_den})")
    out.append(f"    honest nuisance : {pct(h_r)}  ({h_num}/{h_den})")
    out.append("")

    text = "\n".join(out) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text, flush=True)
    print(f"[fc_falsepos_parse] wrote results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
