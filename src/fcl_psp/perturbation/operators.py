"""Window-level measurement-fidelity perturbation operators (Experiment B).

Pure functions on ``(N, L, F)`` float32 window tensors. Each operator:
  * never mutates its input (returns a fresh array);
  * takes an explicit ``numpy.random.Generator`` for all stochasticity;
  * returns an unchanged **copy** at its "clean" level (identity contract), so a
    perturbation sweep that includes the clean level reproduces the reference
    numbers exactly.

Channel addressing is resolved from ``feature_names`` (the authoritative order
from ``load_meta_data``), NOT from ``psp_helper.constants.CURRENT_CHANNELS``:
the actual PROTECT-90 windows order channels currents-first (0-2) then voltages
(3-5) per relay, which is the opposite of those constants. Resolving by the
``_cur_`` / ``_vol_`` tokens in the names is therefore mandatory for correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np

Level = Union[int, float, str, None]

_CLEAN_SENTINELS = {"clean", "no_sat", "none", ""}


def _is_clean(level: Level) -> bool:
    if level is None:
        return True
    if isinstance(level, str):
        return level.strip().lower() in _CLEAN_SENTINELS
    return False


# ---------------------------------------------------------------------------
# Channel layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelLayout:
    """Column groupings derived from ``feature_names`` (length F)."""

    feature_names: List[str]
    current_cols: np.ndarray  # int indices of current channels
    voltage_cols: np.ndarray  # int indices of voltage channels
    relay_of_col: np.ndarray  # (F,) relay index per column
    relay_names: List[str]  # relay prefix per relay index

    @property
    def n_relays(self) -> int:
        return len(self.relay_names)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _relay_prefix(name: str) -> str:
    """Relay identity = the name up to the ``_cur_``/``_vol_`` quantity token.

    e.g. ``Bus_1_Line_01_02A_cur_L1_A`` -> ``Bus_1_Line_01_02A``.
    """
    for token in ("_cur_", "_vol_"):
        idx = name.find(token)
        if idx != -1:
            return name[:idx]
    # Fallback: strip a trailing ``_L<phase>_<unit>`` if present.
    return name.rsplit("_", 2)[0]


def channel_layout(feature_names: List[str]) -> ChannelLayout:
    """Build a :class:`ChannelLayout` from the authoritative feature names."""
    feature_names = list(feature_names)
    current_cols, voltage_cols = [], []
    relay_of_col = np.full(len(feature_names), -1, dtype=int)
    relay_index: dict = {}
    relay_names: List[str] = []

    for i, name in enumerate(feature_names):
        low = name.lower()
        if "_cur_" in low:
            current_cols.append(i)
        elif "_vol_" in low:
            voltage_cols.append(i)
        prefix = _relay_prefix(name)
        if prefix not in relay_index:
            relay_index[prefix] = len(relay_names)
            relay_names.append(prefix)
        relay_of_col[i] = relay_index[prefix]

    return ChannelLayout(
        feature_names=feature_names,
        current_cols=np.asarray(current_cols, dtype=int),
        voltage_cols=np.asarray(voltage_cols, dtype=int),
        relay_of_col=relay_of_col,
        relay_names=relay_names,
    )


# ---------------------------------------------------------------------------
# Additive Gaussian noise (SNR sweep)   [B.1]
# ---------------------------------------------------------------------------
def add_gaussian_snr(x: np.ndarray, snr_db: Level, rng: np.random.Generator) -> np.ndarray:
    """Add per-window, per-channel white Gaussian noise at a target SNR (dB).

    ``std = rms(channel) / 10**(snr_db/20)`` where ``rms`` is computed over the
    time axis per (window, channel). Applied to all channels. ``snr_db`` may be
    ``None`` or ``"clean"`` for the identity level. Formula mirrors
    ``protect90_baselines .../run_experiment.py::_NoisyDataset``.
    """
    if _is_clean(snr_db):
        return x.copy()
    snr = float(snr_db)  # type: ignore[arg-type]
    xd = x.astype(np.float64)
    rms = np.sqrt(np.mean(xd * xd, axis=1, keepdims=True))  # (N,1,F)
    std = rms / (10.0 ** (snr / 20.0))
    noise = rng.standard_normal(xd.shape) * std
    return (xd + noise).astype(x.dtype, copy=False)


# ---------------------------------------------------------------------------
# CT saturation proxy (current channels only)   [B.3]
# ---------------------------------------------------------------------------
def apply_ct_saturation(
    x: np.ndarray,
    c: Level,
    layout: ChannelLayout,
    rng: np.random.Generator,
    *,
    randomize_onset: bool = True,
) -> np.ndarray:
    """Proxy CT saturation: magnitude-clip current channels at ``c * peak``.

    ``c`` is the retained fraction of each current channel's clean peak
    (lower = more severe); ``None``/``"no_sat"`` is the identity level. The
    saturation onset is randomized per window (a time index from which the clip
    applies), emulating a saturation-onset phase per realization. This is an
    explicit *proxy*, not a full instrument-transformer model (cite IEEE
    C37.110); a faithful CT model is future work.
    """
    if _is_clean(c):
        return x.copy()
    frac = float(c)  # type: ignore[arg-type]
    cur = layout.current_cols
    out = x.copy()
    if cur.size == 0:
        return out
    seg = x[:, :, cur].astype(np.float64)  # (N, L, nc)
    peak = np.max(np.abs(seg), axis=1, keepdims=True)  # (N, 1, nc)
    clip_level = frac * peak
    clipped = np.clip(seg, -clip_level, clip_level)
    N, L = x.shape[0], x.shape[1]
    if randomize_onset:
        t0 = rng.integers(0, max(1, L // 2), size=N)  # (N,)
        active = (np.arange(L)[None, :] >= t0[:, None])[:, :, None]  # (N,L,1)
        seg_out = np.where(active, clipped, seg)
    else:
        seg_out = clipped
    out[:, :, cur] = seg_out.astype(x.dtype, copy=False)
    return out


# ---------------------------------------------------------------------------
# Synchronization jitter (per-relay time offset)   [B.4]
# ---------------------------------------------------------------------------
def _shift_time_edge(a: np.ndarray, d: int) -> np.ndarray:
    """Shift ``a`` (N, L, k) along the time axis by ``d`` samples, edge-padded.

    ``d > 0`` delays (forward shift, head padded with the first sample); ``d < 0``
    advances (backward shift, tail padded with the last sample).
    """
    if d == 0:
        return a.copy()
    N, L, k = a.shape
    out = np.empty_like(a)
    if d > 0:
        d = min(d, L)
        out[:, d:, :] = a[:, : L - d, :]
        out[:, :d, :] = a[:, 0:1, :]
    else:
        s = min(-d, L)
        out[:, : L - s, :] = a[:, s:, :]
        out[:, L - s :, :] = a[:, -1:, :]
    return out


def apply_sync_jitter(
    x: np.ndarray,
    delta_max: int,
    layout: ChannelLayout,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-relay random time offset ``d_r ~ U{-delta_max..+delta_max}`` samples.

    All six channels of a relay shift together (edge-padded). ``delta_max == 0``
    is the identity level. Each relay draws its own offset, so the two terminals
    of a line become desynchronized — the effect that stresses two-ended FL and
    multi-relay fusion.
    """
    if delta_max is None or int(delta_max) == 0:
        return x.copy()
    dmax = int(delta_max)
    out = x.copy()
    for r in range(layout.n_relays):
        d = int(rng.integers(-dmax, dmax + 1))
        if d == 0:
            continue
        cols = np.where(layout.relay_of_col == r)[0]
        if cols.size:
            out[:, :, cols] = _shift_time_edge(x[:, :, cols], d)
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def apply_axis(
    axis: str,
    level: Level,
    x: np.ndarray,
    layout: ChannelLayout,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the perturbation for ``axis`` at ``level`` to ``x``."""
    if axis in ("noise", "train_aug"):
        return add_gaussian_snr(x, level, rng)
    if axis == "ct_saturation":
        return apply_ct_saturation(x, level, layout, rng)
    if axis == "jitter":
        return apply_sync_jitter(x, level, layout, rng)  # type: ignore[arg-type]
    raise ValueError(f"Unknown perturbation axis {axis!r}")
