#!/usr/bin/env python3
"""
Export MLflow runs to LaTeX tables.

Examples
--------
# Local file-based backend
python export_mlflow_to_latex.py \
  --tracking-uri file://$PWD/mlruns \
  --experiment default \
  --filter "params.dataset = 'new_double_line_90kv_multi_fault_10k' and tags.completed = 'true'" \
  --order "metrics.mean_rmse ASC" \
  --outdir ./latex_tables

# HTTP tracking server
python export_mlflow_to_latex.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment default \
  --filter "params.dataset = 'new_double_line_90kv_multi_fault_10k' and tags.completed = 'true'"
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


def latex_escape(text: str) -> str:
    """
    Escape LaTeX special characters in text (underscores, %, &, etc.)
    """
    if text is None:
        return ""
    conv = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    regex = re.compile("|".join(re.escape(str(k)) for k in conv.keys()))
    return regex.sub(lambda m: conv[m.group()], str(text))


def human_duration_ms(ms):
    if ms is None:
        return ""
    seconds = int(ms // 1000)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


def fmt_time_ms(ms):
    if ms is None:
        return ""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d")


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


def get_experiment_id(client: MlflowClient, name_or_id: str) -> str:
    # If it's already an ID, return; else resolve by name.
    if name_or_id.isdigit():
        return name_or_id
    exp = client.get_experiment_by_name(name_or_id)
    if exp is None:
        raise ValueError(f"Experiment '{name_or_id}' not found.")
    return exp.experiment_id


def collect_runs(
    client: MlflowClient,
    experiment_id: str,
    filter_string: Optional[str],
    order_by: Optional[str],
    max_results: int = 5000,
) -> List[Any]:
    order_list = [order_by] if order_by else None
    return client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_list,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    )


def runs_to_dataframe(runs: List[Any]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r.data.metrics
        p = r.data.params
        rows.append(
            {
                "Model": latex_escape(p.get("model", "")),
                "$d$": p.get("n_features", ""),
                "$\\overline{\\mathrm{RMSE}}$": (
                    f"{m.get('mean_rmse', float('nan')):.4f}" if "mean_rmse" in m else ""
                ),
                "$\\sigma_{\\mathrm{RMSE}}$": (
                    f"{m.get('std_rmse', float('nan')):.4f}" if "std_rmse" in m else ""
                ),
                "$\\overline{R^2}$": (
                    f"{m.get('mean_r2', float('nan')):.4f}" if "mean_r2" in m else ""
                ),
                "$\\sigma_{R^2}$": (
                    f"{m.get('std_r2', float('nan')):.4f}" if "std_r2" in m else ""
                ),
                "Date": fmt_time_ms(r.info.start_time),
                "Train Time": human_duration_ms(
                    r.info.end_time - r.info.start_time if r.info.end_time else None
                ),
            }
        )
    return pd.DataFrame(rows)


def format_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    # choose columns that exist; prioritize regression metrics if present else classification
    base_cols = ["model", "n_features", "created", "duration"]
    reg_cols = ["mean_rmse", "std_rmse", "mean_r2", "std_r2", "mean_mae", "std_mae"]
    cls_cols = ["acc", "f1"]

    cols = base_cols.copy()
    if df[reg_cols].notna().any().any():
        cols = [
            "model",
            "n_features",
            "mean_rmse",
            "std_rmse",
            "mean_r2",
            "std_r2",
            "created",
            "duration",
        ]
    elif df[cls_cols].notna().any().any():
        cols = ["model", "n_features", "acc", "f1", "created", "duration"]

    out = df[cols].copy()

    # numeric formatting
    for c in [
        "mean_rmse",
        "std_rmse",
        "mean_r2",
        "std_r2",
        "mean_mae",
        "std_mae",
        "acc",
        "f1",
    ]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    # rename nicer headers for LaTeX
    rename_map = {
        "model": "Model",
        "n_features": "$d$",
        "mean_rmse": "$\\overline{\\mathrm{RMSE}}$",
        "std_rmse": "$\\sigma_{\\mathrm{RMSE}}$",
        "mean_r2": "$\\overline{R^2}$",
        "std_r2": "$\\sigma_{R^2}$",
        "mean_mae": "$\\overline{\\mathrm{MAE}}$",
        "std_mae": "$\\sigma_{\\mathrm{MAE}}$",
        "acc": "Acc.",
        "f1": "F1",
        "created": "Date",
        "duration": "Train Time",
    }
    out = out.rename(columns=rename_map)
    return out


def df_to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    colfmt = "l" + "r" * (len(df.columns) - 1)
    return df.to_latex(
        index=False,
        escape=False,  # we escaped manually
        caption=caption,
        label=label,
        column_format=colfmt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--order", default="metrics.mean_rmse ASC")
    ap.add_argument("--outdir", default="./latex_tables")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name(args.experiment)
    runs = client.search_runs(
        [exp.experiment_id], filter_string=args.filter, order_by=[args.order]
    )

    df = runs_to_dataframe(runs)

    caption = f"Results for {latex_escape(args.experiment)}"
    label = "tab:" + re.sub(r"[^a-zA-Z0-9]+", "_", args.experiment)

    tex = df_to_latex_table(df, caption, label)

    Path(args.outdir).mkdir(exist_ok=True)
    # use sanitized filename
    fname = sanitize_filename(label) + ".tex"
    outpath = Path(args.outdir) / fname
    outpath.write_text(tex, encoding="utf-8")

    print(f"LaTeX table written to {outpath.resolve()}")
    print(tex)


if __name__ == "__main__":
    main()
