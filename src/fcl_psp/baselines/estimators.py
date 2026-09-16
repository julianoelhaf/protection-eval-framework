"""Conventional protection-baseline estimators under the framework protocol.

Three estimators, all consuming the raw ``(N, L, F)`` windows (physical volts/amps,
no standardization) and the authoritative ``feature_names``:

* :class:`ConventionalFaultClassifier` — symmetrical-component / superimposed
  phase selector. Pickup thresholds ``{tau_p, tau_g}`` are **fit on training
  folds only** (grid search maximizing train macro-F1), then frozen — the Step-6
  fitting boundary even a "no-training" method has. Predicts the same integer
  ``fault_class`` IDs as the ML FC models (via ``FAULT_LABEL_TO_ID``).
* :class:`TwoEndedFaultLocator` — synchronized two-ended positive-sequence
  distance (relay-pair observability). No fit.
* :class:`SingleEndedFaultLocator` — one-ended reactance-to-fault with
  ground-truth loop selection (single-relay observability). No fit.

FL predictions are in % of line length from the lower-index bus (the ML FL
target convention).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from psp_helper.constants import FAULT_LABEL_TO_ID
from sklearn.metrics import f1_score

from fcl_psp.baselines.algorithms.fault_location import locate_two_ended
from fcl_psp.baselines.algorithms.single_ended import single_ended_reactance
from fcl_psp.baselines.data.line_registry import LineRegistry
from fcl_psp.baselines.dsp.phasor import fundamental_phasors
from fcl_psp.baselines.dsp.relay_features import relay_features
from fcl_psp.baselines.prefault import episode_baselines

_EPS = 1e-6
_NO_FAULT_ID = FAULT_LABEL_TO_ID["no_fault"]


# ---------------------------------------------------------------------------
# phase set + ground -> framework fault_class id (matches build_fault_label)
# ---------------------------------------------------------------------------
def phases_ground_to_id(phases: Sequence[int], grounded: bool) -> int:
    """Map a selected phase set + ground flag to a framework ``fault_class`` ID.

    Reconstructs the exact ``build_fault_label`` string
    (``{event_type}_{ABC}{G}``) and looks it up in ``FAULT_LABEL_TO_ID`` so the
    conventional FC classes coincide with the ML FC classes. Unknown/degenerate
    combinations fall back to ``no_fault``.
    """
    ph = sorted(set(int(p) for p in phases))
    if not ph:
        return _NO_FAULT_ID
    a, b, c = (0 in ph), (1 in ph), (2 in ph)
    n = len(ph)
    if n == 1:
        event_type, g = "flt_1phg_shc", True  # single-phase -> SLG
    elif n == 2:
        event_type, g = ("flt_2phg_shc", True) if grounded else ("flt_2ph_shc", False)
    else:
        event_type, g = "flt_3ph_shc", False  # 3-phase symmetric
    phases_str = ("A" if a else "") + ("B" if b else "") + ("C" if c else "")
    label = f"{event_type}_{phases_str}{'G' if g else ''}"
    return FAULT_LABEL_TO_ID.get(label, _NO_FAULT_ID)


def _code_to_id_table() -> np.ndarray:
    """(16,) lookup: bit-code (a|b<<1|c<<2|g<<3) -> fault_class id."""
    table = np.zeros(16, dtype=int)
    for code in range(16):
        phases = [p for p in range(3) if code & (1 << p)]
        grounded = bool(code & 8)
        table[code] = phases_ground_to_id(phases, grounded)
    return table


# ---------------------------------------------------------------------------
# Fault classification — symmetrical-component phase selector
# ---------------------------------------------------------------------------
class ConventionalFaultClassifier:
    def __init__(
        self,
        registry: LineRegistry,
        samples_per_cycle: int,
        pickup_grid: Sequence[float],
        ground_grid: Sequence[float],
    ):
        self.registry = registry
        self.relays = registry.relays()
        self.spc = int(samples_per_cycle)
        self.pickup_grid = list(pickup_grid)
        self.ground_grid = list(ground_grid)
        self._code_to_id = _code_to_id_table()
        self.tau_p_: Optional[float] = None
        self.tau_g_: Optional[float] = None

    def precompute(self, windows: np.ndarray, labels: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Stream over the memmap once -> compact per-window feature table.

        For each window: choose the faulted relay (max summed current-rise over
        the episode pre-fault baseline), and store that relay's phase current
        magnitudes, its pre-fault baseline, its residual |3 I0|, and |I1|.
        Memory: ~O(N) floats, so this is safe even for the full ~261k-window FC set.
        """
        baselines = episode_baselines(windows, labels, self.relays, self.spc)
        sample_ids = labels["sample_id"].to_numpy()
        n = len(labels)
        i_mag = np.zeros((n, 3), dtype=np.float64)
        base = np.zeros((n, 3), dtype=np.float64)
        i_res = np.zeros(n, dtype=np.float64)
        i1_abs = np.zeros(n, dtype=np.float64)
        for row in range(n):
            sid = int(sample_ids[row])
            b = baselines[sid]  # (n_relays, 3)
            feats = relay_features(windows[row], self.relays, self.spc)
            imag = np.stack([f.i_mag for f in feats])  # (n_relays, 3)
            rise = np.sum(np.maximum(imag - b, 0.0), axis=1)  # (n_relays,)
            r = int(np.argmax(rise))
            i_mag[row] = imag[r]
            base[row] = b[r]
            i_res[row] = abs(feats[r].i_residual)
            i1_abs[row] = abs(feats[r].i1)
        return {"i_mag": i_mag, "base": base, "i_res": i_res, "i1_abs": i1_abs}

    def predict_ids(self, feat: Dict[str, np.ndarray], tau_p: float, tau_g: float) -> np.ndarray:
        mask = feat["i_mag"] > tau_p * np.maximum(feat["base"], _EPS)  # (N,3)
        grounded = feat["i_res"] > tau_g * np.maximum(feat["i1_abs"], _EPS)  # (N,)
        code = (
            mask[:, 0].astype(int)
            + 2 * mask[:, 1].astype(int)
            + 4 * mask[:, 2].astype(int)
            + 8 * grounded.astype(int)
        )
        return self._code_to_id[code]

    def fit(self, feat_train: Dict[str, np.ndarray], y_train: np.ndarray) -> dict:
        best = None
        for tau_p in self.pickup_grid:
            for tau_g in self.ground_grid:
                pred = self.predict_ids(feat_train, tau_p, tau_g)
                f1 = f1_score(y_train, pred, average="macro", zero_division=0)
                if best is None or f1 > best["f1"]:
                    best = {"f1": float(f1), "tau_p": float(tau_p), "tau_g": float(tau_g)}
        self.tau_p_, self.tau_g_ = best["tau_p"], best["tau_g"]
        return best

    def predict(self, feat: Dict[str, np.ndarray]) -> np.ndarray:
        if self.tau_p_ is None:
            raise RuntimeError("call fit() first")
        return self.predict_ids(feat, self.tau_p_, self.tau_g_)


# ---------------------------------------------------------------------------
# Fault localization — impedance-based (no fit)
# ---------------------------------------------------------------------------
def _canon_line(value: str, known: List[str]) -> str:
    """Normalize a ``y_fault_line`` value to a canonical ``Line_f_t_x`` name."""
    s = str(value)
    if s in known:
        return s
    low = s.lower()
    if low in known:
        return low
    # e.g. "Line_1_2_A" -> "Line_1_2_a"
    parts = s.split("_")
    if len(parts) >= 4 and parts[0].lower() == "line":
        cand = f"Line_{parts[1]}_{parts[2]}_{parts[3].lower()}"
        if cand in known:
            return cand
    raise KeyError(f"cannot map fault line {value!r} to one of {known}")


def faulted_phase_current_mag(windows, labels, registry, spc: int) -> np.ndarray:
    """Per-window settledness proxy = largest faulted-phase current magnitude at S.

    A conventional distance/location element latches its estimate at reliable fault
    current. Onset-conditioned windows (Eq. 5) include many where the fault has only
    just begun inside the window and the current has not yet risen — those give
    unreliable impedance estimates. This magnitude lets the evaluation pick, per
    episode, the settled window a relay would actually use.
    """
    known = registry.line_names
    lines = labels["y_fault_line"].astype(str).to_numpy()
    pa = labels["y_phase_A"].to_numpy()
    pb = labels["y_phase_B"].to_numpy()
    pc = labels["y_phase_C"].to_numpy()
    out = np.zeros(len(labels), dtype=np.float64)
    for row in range(len(labels)):
        term = registry.terminal(_canon_line(lines[row], known))
        i_abc = fundamental_phasors(windows[row][:, term.s_cur], spc)
        fp = [p for p, f in enumerate((pa[row], pb[row], pc[row])) if bool(f)]
        out[row] = max(abs(i_abc[p]) for p in fp) if fp else float(np.max(np.abs(i_abc)))
    return out


class TwoEndedFaultLocator:
    """Synchronized two-ended positive-sequence distance (% line from lower bus)."""

    def __init__(
        self,
        registry: LineRegistry,
        samples_per_cycle: int,
        *,
        current_sign: int = 1,
        min_current_ratio: float = 1e-2,
    ):
        self.registry = registry
        self.spc = int(samples_per_cycle)
        self.current_sign = int(current_sign)
        self.min_current_ratio = float(min_current_ratio)

    def predict(
        self, windows: np.ndarray, labels: pd.DataFrame, params_df: pd.DataFrame
    ) -> np.ndarray:
        known = self.registry.line_names
        sids = labels["sample_id"].to_numpy()
        lines = labels["y_fault_line"].astype(str).to_numpy()
        preds = np.full(len(labels), np.nan, dtype=np.float64)
        for row in range(len(labels)):
            line = _canon_line(lines[row], known)
            term = self.registry.terminal(line)
            params = self.registry.params(params_df, int(sids[row]), line)
            m = locate_two_ended(
                windows[row],
                term,
                params,
                self.spc,
                current_sign=self.current_sign,
                min_current_ratio=self.min_current_ratio,
            )
            preds[row] = 100.0 * m
        return preds


class SingleEndedFaultLocator:
    """One-ended reactance-to-fault, ground-truth loop selection (% line from S)."""

    def __init__(
        self,
        registry: LineRegistry,
        samples_per_cycle: int,
        *,
        neglect_mutual: bool = True,
        terminal: str = "S",
    ):
        self.registry = registry
        self.spc = int(samples_per_cycle)
        self.neglect_mutual = bool(neglect_mutual)
        self.terminal = terminal  # "S" (lower bus, distance reference) or "R"

    def predict(
        self, windows: np.ndarray, labels: pd.DataFrame, params_df: pd.DataFrame
    ) -> np.ndarray:
        known = self.registry.line_names
        sids = labels["sample_id"].to_numpy()
        lines = labels["y_fault_line"].astype(str).to_numpy()
        pa = labels["y_phase_A"].to_numpy()
        pb = labels["y_phase_B"].to_numpy()
        pc = labels["y_phase_C"].to_numpy()
        grd = labels["y_is_grounded"].to_numpy()
        preds = np.full(len(labels), np.nan, dtype=np.float64)
        for row in range(len(labels)):
            line = _canon_line(lines[row], known)
            term = self.registry.terminal(line)
            params = self.registry.params(params_df, int(sids[row]), line)
            v_idx = term.s_vol if self.terminal == "S" else term.r_vol
            i_idx = term.s_cur if self.terminal == "S" else term.r_cur
            v_abc = fundamental_phasors(windows[row][:, v_idx], self.spc)
            i_abc = fundamental_phasors(windows[row][:, i_idx], self.spc)
            faulted_phases = [p for p, f in enumerate((pa[row], pb[row], pc[row])) if bool(f)]
            m, _mode = single_ended_reactance(
                v_abc,
                i_abc,
                params.z1_total,
                faulted_phases,
                bool(grd[row]),
                z0_line=params.z0_total,
                neglect_mutual=self.neglect_mutual,
            )
            preds[row] = 100.0 * m
        return preds
