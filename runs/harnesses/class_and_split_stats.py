"""Class distribution + per-fold test-set sizes for the case study (labels only).

Answers reviewer requests for (a) the class balance behind the imbalanced-11-class
macro-F1 metric and (b) the test-set size behind each reported cell. Loads only the
label table + the GroupKFold split (no window data touched); runs in seconds.

    python runs/harnesses/class_and_split_stats.py
"""
import os
import numpy as np
from collections import Counter
from hydra import compose, initialize_config_dir
from sklearn.model_selection import GroupKFold

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")
WINDIR = "/path/to/datasets/PROTECT-90/windows/PROTECT-90"


def load(target, w):
    with initialize_config_dir(config_dir=CFG, version_base=None):
        cfg = compose(config_name="main-config", overrides=[
            f"window_extraction.windows_local_dir={WINDIR}",
            "tracking.mode=disabled",
            f"training.target_label={target}",
            f"window_extraction.window_length={w}"])
    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets
    from psp_helper.windows_helper import load_windows_and_labels
    _, labels, ri = load_windows_and_labels(cfg)
    sids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    return cfg, np.asarray(sids), np.asarray(y)


def main():
    from psp_helper.constants import FAULT_LABEL_TO_ID
    id2lab = {v: k for k, v in FAULT_LABEL_TO_ID.items()}
    cfg, sids, y = load("event_type", 0.020)
    N = len(y); cnt = Counter(y.tolist())
    print(f"FC class distribution (event_type, 20ms) - N={N} windows, {len(set(sids))} episodes")
    for cid in sorted(cnt):
        print(f"  {str(id2lab.get(cid, cid)):16s} id={cid:2d} n={cnt[cid]:6d} {100*cnt[cid]/N:5.2f}%")
    nf = FAULT_LABEL_TO_ID.get("no_fault")
    if nf in cnt:
        print(f"  no-fault fraction = {100*cnt[nf]/N:.2f}%")
    print(f"  imbalance (max/min class) = {max(cnt.values())/min(cnt.values()):.1f}x")

    def foldsizes(tag, sids, y):
        gkf = GroupKFold(n_splits=int(cfg.training.n_splits))
        print(f"\ntest-set sizes {tag} (5-fold GroupKFold) - N={len(y)}")
        for k, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups=sids), 1):
            print(f"  fold {k}: {len(te):6d} windows / {len(set(sids[te])):5d} episodes")

    foldsizes("FC 20ms (all windows)", sids, y)
    _, sfll, yfl = load("y_fault_location", 0.020)
    foldsizes("FL 20ms (fault-only)", sfll, yfl)


if __name__ == "__main__":
    main()
