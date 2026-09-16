# Vendored from protect-90-baselines (package protect90_baselines), data/channel_map.py.
# Source repo: /path/to/repos/protect-90-baselines (pure-numpy; unit-tested there).
# Do not edit logic here without syncing upstream. Imports rewired to fcl_psp.baselines.*.

"""Parse the 48 PROTECT-90 channel names into a structured channel map.

Each feature name looks like ``Bus_2_Line_02_03A_cur_L1_A`` =
``Bus_{bus}_Line_{from:02d}_{to:02d}{A|B}_{cur|vol}_L{phase}_{A|V}``. We parse the
names rather than hardcoding indices, so a re-ordering of the release is caught
instead of silently mislabelled.

Canonical line name: ``Line_{from}_{to}_{a|b}`` (e.g. ``Line_2_3_a``). The
sending terminal S = the lower-numbered bus (= ``line_from``, the reference for
the fault-location distance); the receiving terminal R = the higher-numbered bus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import schema

_NAME_RE = re.compile(
    r"^Bus_(?P<bus>\d+)_Line_(?P<f>\d+)_(?P<t>\d+)(?P<sfx>[AB])_"
    r"(?P<q>cur|vol)_L(?P<ph>[123])_(?P<unit>[AV])$"
)


@dataclass
class LineTerminals:
    """Channel indices for one line section's two terminals."""

    line: str
    s_bus: int
    r_bus: int
    s_cur: list[int] = field(default_factory=list)  # [L1, L2, L3]
    s_vol: list[int] = field(default_factory=list)
    r_cur: list[int] = field(default_factory=list)
    r_vol: list[int] = field(default_factory=list)

    def is_complete(self) -> bool:
        return all(len(v) == 3 for v in (self.s_cur, self.s_vol, self.r_cur, self.r_vol))


def canonical_line(f: int, t: int, suffix: str) -> str:
    return f"Line_{f}_{t}_{suffix.lower()}"


def parse_feature_names(feature_names: list[str]) -> dict[str, LineTerminals]:
    """Build ``{canonical_line: LineTerminals}`` from the 48 feature names."""
    if len(feature_names) != schema.N_CHANNELS:
        raise schema.SchemaError(
            f"expected {schema.N_CHANNELS} feature names, got {len(feature_names)}"
        )

    # First pass: discover the buses bounding each line section.
    buses_per_line: dict[str, set] = {}
    parsed = []
    for idx, name in enumerate(feature_names):
        m = _NAME_RE.match(name)
        if not m:
            raise schema.SchemaError(f"unparseable channel name at {idx}: {name!r}")
        f, t, sfx = int(m["f"]), int(m["t"]), m["sfx"]
        line = canonical_line(f, t, sfx)
        bus = int(m["bus"])
        buses_per_line.setdefault(line, set()).add(bus)
        parsed.append((idx, line, bus, m["q"], int(m["ph"])))

    terminals: dict[str, LineTerminals] = {}
    for line, buses in buses_per_line.items():
        if len(buses) != 2:
            raise schema.SchemaError(
                f"line {line} should have exactly 2 terminals, found buses {sorted(buses)}"
            )
        s_bus, r_bus = min(buses), max(buses)
        terminals[line] = LineTerminals(line=line, s_bus=s_bus, r_bus=r_bus)

    # Second pass: assign each channel to S or R terminal, ordered by phase.
    tmp: dict[str, dict] = {
        line: {"s_cur": {}, "s_vol": {}, "r_cur": {}, "r_vol": {}} for line in terminals
    }
    for idx, line, bus, q, ph in parsed:
        term = "s" if bus == terminals[line].s_bus else "r"
        key = f"{term}_{q[:3] if q == 'cur' else 'vol'}"
        tmp[line][key][ph] = idx

    for line, lt in terminals.items():
        lt.s_cur = [tmp[line]["s_cur"][p] for p in (1, 2, 3)]
        lt.s_vol = [tmp[line]["s_vol"][p] for p in (1, 2, 3)]
        lt.r_cur = [tmp[line]["r_cur"][p] for p in (1, 2, 3)]
        lt.r_vol = [tmp[line]["r_vol"][p] for p in (1, 2, 3)]
        if not lt.is_complete():
            raise schema.SchemaError(f"line {line} has incomplete channel set")

    return terminals
