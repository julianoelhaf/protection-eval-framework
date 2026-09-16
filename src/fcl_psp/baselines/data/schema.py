# Vendored from protect-90-baselines (package protect90_baselines), data/schema.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Dataset schema constants and validation for PROTECT-90 (windowed memmap).

These constants encode the *expected* structure of the released windowed tensors
and label tables. They are used to fail loudly on a schema mismatch (missing
columns, wrong channel count, etc.) rather than silently mislabelling.
"""

from __future__ import annotations

from collections.abc import Iterable

# 48 measurement channels = 8 relays x 6 (3 phase currents L1/L2/L3 then 3 phase
# voltages L1/L2/L3). Confirmed from the window meta.json feature_names.
N_CHANNELS = 48
N_RELAYS = 8
CHANNELS_PER_RELAY = 6

# Label columns required in the per-window parquet (y_*.parquet).
REQUIRED_Y_COLUMNS: list[str] = [
    "sample_id",
    "status",
    "y_fault_present",
    "y_fault_line",
    "y_fault_location",
    "y_is_grounded",
    "y_phase_A",
    "y_phase_B",
    "y_phase_C",
    "window_start_time",
]

# Status values in the per-window parquet.
STATUS_VALUES = ("clean", "fault_start", "in_fault")

# The four protected line sections.
LINE_NAMES: list[str] = ["Line_1_2_a", "Line_1_2_b", "Line_2_3_a", "Line_2_3_b"]

# CSV prefix per line section (scenario metadata column prefix).
LINE_CSV_PREFIX = {
    "Line_1_2_a": "line_1_2_a",
    "Line_1_2_b": "line_1_2_b",
    "Line_2_3_a": "line_2_3_a",
    "Line_2_3_b": "line_2_3_b",
}

# Per-line scenario-CSV columns (suffixes appended to the prefix).
LINE_PARAM_SUFFIXES = (
    "length",  # km
    "xline",  # positive-seq X' (Ohm/km)
    "rline",  # positive-seq R' (Ohm/km)
    "cline",  # positive-seq C' (uF/km)
    "xline0",  # zero-seq X0' (Ohm/km)
    "rline0",  # zero-seq R0' (Ohm/km)
    "cline0",  # zero-seq C0' (uF/km)
)

# Required non-per-line scenario-CSV columns.
REQUIRED_CSV_COLUMNS: list[str] = [
    "sample_id",
    "sc_type",
    "sc_location",
    "phase_select",
    "fault_target",
    "fault_resistance",
]


class SchemaError(ValueError):
    """Raised when the dataset does not match the expected schema."""


def validate_y_columns(columns: Iterable[str]) -> None:
    missing = [c for c in REQUIRED_Y_COLUMNS if c not in columns]
    if missing:
        raise SchemaError(f"y parquet missing required columns: {missing}")


def validate_csv_columns(columns: Iterable[str]) -> None:
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in columns]
    for prefix in LINE_CSV_PREFIX.values():
        for suf in LINE_PARAM_SUFFIXES:
            col = f"{prefix}_{suf}"
            if col not in columns:
                missing.append(col)
    if missing:
        raise SchemaError(f"scenario CSV missing required columns: {missing}")


def validate_channel_count(n: int) -> None:
    if n != N_CHANNELS:
        raise SchemaError(f"expected {N_CHANNELS} channels, got {n}")
