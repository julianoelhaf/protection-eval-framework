"""Per-episode pre-fault magnitude baselines for the conventional phase selector.

Framework-native replacement for ``protect90_baselines.evaluation.prefault`` (which
couples to that repo's dataset loader). Takes the **first ``status=="clean"``
window per episode** and returns each relay's pre-fault current-magnitude vector,
used by :func:`fault_classification.select_faulted_phases` as the pickup baseline.

If an episode has no clean window in the provided set (e.g. the onset-only FL
subset), it falls back to the per-relay minimum current magnitude across that
episode's windows — a conservative pre-fault proxy.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from fcl_psp.baselines.dsp.relay_features import relay_features


def episode_baselines(
    windows: np.ndarray,
    labels: pd.DataFrame,
    relays: List[dict],
    samples_per_cycle: int,
    *,
    status_col: str = "status",
    id_col: str = "sample_id",
    clean_value: str = "clean",
) -> Dict[int, np.ndarray]:
    """Return ``{sample_id: i_mag_baseline}`` of shape ``(n_relays, 3)``.

    ``windows`` is ``(N, L, F)`` aligned row-for-row with ``labels``.
    """
    status = labels[status_col].astype(str).to_numpy() if status_col in labels else None
    sample_ids = labels[id_col].to_numpy()
    n_relays = len(relays)

    baselines: Dict[int, np.ndarray] = {}
    for sid in np.unique(sample_ids):
        ep_rows = np.where(sample_ids == sid)[0]
        clean_rows = (
            ep_rows[status[ep_rows] == clean_value] if status is not None else np.array([], int)
        )
        if clean_rows.size:
            row = int(clean_rows[0])
            feats = relay_features(windows[row], relays, samples_per_cycle)
            baselines[int(sid)] = np.stack([f.i_mag for f in feats])  # (n_relays, 3)
        else:
            # Fallback: per-relay, per-phase minimum |I| across the episode's windows.
            mins = np.full((n_relays, 3), np.inf)
            for row in ep_rows:
                feats = relay_features(windows[int(row)], relays, samples_per_cycle)
                mins = np.minimum(mins, np.stack([f.i_mag for f in feats]))
            baselines[int(sid)] = mins
    return baselines
