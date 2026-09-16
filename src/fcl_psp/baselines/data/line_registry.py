# Vendored from protect-90-baselines (package protect90_baselines), data/line_registry.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Line registry: the four protected line sections and their parameters.

Combines the channel-terminal map (from the window ``feature_names``) with the
per-episode true line parameters from the scenario CSV. The registry is the
single source of truth for "which channels belong to which line" and "what is the
series impedance of the faulted line in episode N".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import schema
from .channel_map import LineTerminals, parse_feature_names


@dataclass(frozen=True)
class LineParams:
    """True series parameters of one line section in one episode."""

    sample_id: int
    line: str
    length_km: float
    r1_ohm_per_km: float
    x1_ohm_per_km: float
    r0_ohm_per_km: float
    x0_ohm_per_km: float

    @property
    def z1_total(self) -> complex:
        return complex(self.r1_ohm_per_km, self.x1_ohm_per_km) * self.length_km

    @property
    def z0_total(self) -> complex:
        return complex(self.r0_ohm_per_km, self.x0_ohm_per_km) * self.length_km


class LineRegistry:
    """Channel terminals (from feature_names) + parameter access (from CSV)."""

    def __init__(self, feature_names: list[str]):
        self.terminals: dict[str, LineTerminals] = parse_feature_names(feature_names)
        # Verify the discovered lines are exactly the expected four.
        found = set(self.terminals)
        expected = set(schema.LINE_NAMES)
        if found != expected:
            raise schema.SchemaError(
                f"line registry mismatch: found {sorted(found)}, expected {sorted(expected)}"
            )

    @property
    def line_names(self) -> list[str]:
        return list(schema.LINE_NAMES)

    def terminal(self, line: str) -> LineTerminals:
        if line not in self.terminals:
            raise KeyError(f"unknown line {line!r}; known: {self.line_names}")
        return self.terminals[line]

    def params(self, params_df: pd.DataFrame, sample_id: int, line: str) -> LineParams:
        """True parameters of ``line`` in episode ``sample_id`` from the CSV."""
        if line not in schema.LINE_CSV_PREFIX:
            raise KeyError(f"unknown line {line!r}")
        if sample_id not in params_df.index:
            raise KeyError(f"sample_id {sample_id} not in scenario CSV")
        p = schema.LINE_CSV_PREFIX[line]
        row = params_df.loc[sample_id]
        length = float(row[f"{p}_length"])
        if not length > 0:
            raise ValueError(f"non-positive line length for {line} ep {sample_id}")
        return LineParams(
            sample_id=int(sample_id),
            line=line,
            length_km=length,
            r1_ohm_per_km=float(row[f"{p}_rline"]),
            x1_ohm_per_km=float(row[f"{p}_xline"]),
            r0_ohm_per_km=float(row[f"{p}_rline0"]),
            x0_ohm_per_km=float(row[f"{p}_xline0"]),
        )

    # Parallel double-circuit partner (same bus pair, other circuit suffix).
    _PARALLEL = {
        "Line_1_2_a": "Line_1_2_b",
        "Line_1_2_b": "Line_1_2_a",
        "Line_2_3_a": "Line_2_3_b",
        "Line_2_3_b": "Line_2_3_a",
    }

    def parallel_line(self, line: str) -> str:
        """The parallel circuit sharing this line's bus pair."""
        if line not in self._PARALLEL:
            raise KeyError(f"no parallel circuit known for {line!r}")
        return self._PARALLEL[line]

    def parallel_terminal_cur(self, line: str, terminal: str) -> list[int]:
        """Current channels of the parallel circuit at the *same* bus/terminal.

        Used for the zero-sequence mutual-coupling compensation of the
        single-ended ground-fault loop (the parallel circuit's residual current).
        """
        t = self.terminal(self.parallel_line(line))
        if terminal == "S":
            return t.s_cur
        if terminal == "R":
            return t.r_cur
        raise ValueError(f"terminal must be 'S' or 'R', got {terminal!r}")

    def relays(self) -> list[dict]:
        """The 8 relay measurement points (one per line terminal).

        Each entry: ``{name, line, terminal, cur, vol}`` where ``cur``/``vol`` are
        the 3 phase channel indices. Order follows ``LINE_NAMES`` then S, R.
        """
        out = []
        for line in self.line_names:
            t = self.terminals[line]
            out.append(
                {
                    "name": f"{line}__S",
                    "line": line,
                    "terminal": "S",
                    "cur": t.s_cur,
                    "vol": t.s_vol,
                }
            )
            out.append(
                {
                    "name": f"{line}__R",
                    "line": line,
                    "terminal": "R",
                    "cur": t.r_cur,
                    "vol": t.r_vol,
                }
            )
        return out

    def summary(self) -> list[dict]:
        rows = []
        for name in self.line_names:
            t = self.terminals[name]
            rows.append(
                {
                    "line": name,
                    "s_bus": t.s_bus,
                    "r_bus": t.r_bus,
                    "s_cur": t.s_cur,
                    "s_vol": t.s_vol,
                    "r_cur": t.r_cur,
                    "r_vol": t.r_vol,
                }
            )
        return rows
