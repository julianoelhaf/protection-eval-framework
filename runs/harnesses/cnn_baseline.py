"""1D-CNN modern learned baseline, run through the IDENTICAL evaluation pipeline.

Answers the review finding that no modern learned (deep) baseline is reported: the
strongest learner elsewhere is a shallow MLP on flattened windows. This is a fixed,
untuned, representative 1D-CNN over the window (F=48 channels x L samples), trained
under the SAME episode-grouped 5-fold GroupKFold, per-fold train-only StandardScaler
(fit on the flattened features, exactly as the sklearn models see them), onset-only
FL validity, and the SAME metrics (macro-F1 for FC, MAE %line for FL). Reported as
mean +/- std over the 5 folds, matching every other table.

Architecture (fixed a priori, NOT tuned): 3x [Conv1d(k=5,pad=2)->BN->ReLU->MaxPool(2)]
with channel widths 48->64->128->128, global average pool, linear head. Adam lr 1e-3,
batch 256, <=50 epochs, early stop (patience 8) on an episode-grouped 10% train-internal
validation split. CrossEntropy (FC) / MSE (FL).

Run in the py_dl conda env (has torch+cu, sklearn, hydra, psp_helper):
    /path/to/conda/envs/py_dl/bin/python runs/harnesses/cnn_baseline.py <tag> <target> <wl>
"""
import os
import sys
import time

import numpy as np
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")
WINDIR = os.environ.get("WINDIR", "/path/to/datasets/PROTECT-90/windows/PROTECT-90")
OUT = os.environ.get("CNN_OUT", "/path/to/scratch/cnn_results.txt")
SEED = int(os.environ.get("CNN_SEED", "42"))
SMOKE = os.environ.get("CNN_SMOKE", "0") == "1"   # 1 fold, subsampled, 3 epochs (validation only)


def p(tag, stage, **kv):
    print(("[%s] %s %s" % (tag, stage, " ".join("%s=%s" % kv_ for kv_ in kv.items()))).rstrip(), flush=True)


def log(line):
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


class CNN1D(nn.Module):
    def __init__(self, n_channels, n_out):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 5, padding=2), nn.BatchNorm1d(co),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.features = nn.Sequential(blk(n_channels, 64), blk(64, 128), blk(128, 128))
        self.head = nn.Linear(128, n_out)

    def forward(self, x):                       # x: (B, C, L)
        x = self.features(x)
        x = x.mean(dim=2)                        # global average pool over time
        return self.head(x)


def scale_reshape(windows, idx, sc, L, F, fit):
    X = np.asarray(windows[idx]).reshape(len(idx), L * F)
    X = sc.fit_transform(X) if fit else sc.transform(X)
    return X.reshape(len(idx), L, F).transpose(0, 2, 1).copy()   # (n, F, L)


def train_fold(Xtr, ytr, Xte, task, sids_tr, device):
    n_out = (int(ytr.max()) + 1) if task == "clf" else 1
    # episode-grouped internal val split for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_i, va_i = next(gss.split(Xtr, ytr, groups=sids_tr))
    yt = torch.tensor(ytr, dtype=(torch.long if task == "clf" else torch.float32))
    ds_tr = TensorDataset(torch.tensor(Xtr[tr_i]), yt[tr_i])
    ds_va = TensorDataset(torch.tensor(Xtr[va_i]), yt[va_i])
    dl_tr = DataLoader(ds_tr, batch_size=256, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False, num_workers=0)
    model = CNN1D(Xtr.shape[1], n_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()
    best_val, best_state, patience, wait = np.inf, None, 8, 0
    for epoch in range(3 if SMOKE else 50):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb) if task == "clf" else lossf(out.squeeze(1), yb)
            loss.backward(); opt.step()
        model.eval(); vloss = 0.0; nb = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vloss += (lossf(out, yb) if task == "clf" else lossf(out.squeeze(1), yb)).item(); nb += 1
        vloss /= max(nb, 1)
        if vloss < best_val - 1e-5:
            best_val, best_state, wait = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            xb = torch.tensor(Xte[i:i + 512]).to(device)
            out = model(xb)
            preds.append((out.argmax(1) if task == "clf" else out.squeeze(1)).cpu().numpy())
    return np.concatenate(preds), epoch + 1


def main():
    tag, target, wl = sys.argv[1:4]
    t_start = time.time()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="main-config", overrides=[
            f"training.target_label={target}",
            f"window_extraction.window_length={wl}",
            f"window_extraction.windows_local_dir={WINDIR}",
            "tracking.mode=disabled"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    p(tag, "RUN", target=target, wl=wl, device=device, torch=torch.__version__, seed=SEED)

    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets, write_filtered_memmap
    from psp_helper.windows_helper import load_windows_and_labels
    windows, labels, ri = load_windows_and_labels(cfg)
    if ri is not None:
        windows = write_filtered_memmap(windows, ri, os.path.join(WINDIR, f"X_fo_cnn_{tag}.raw"))
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    sample_ids = np.asarray(sample_ids)
    y = np.asarray(y)
    task = "clf" if target == "event_type" else "reg"
    N, L, F = windows.shape
    p(tag, "DATA", N=N, L=L, F=F, task=task)

    gkf = GroupKFold(n_splits=int(cfg.training.n_splits))
    scores = []
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(N), y, groups=sample_ids), 1):
        if SMOKE:
            tr, te = tr[:3000], te[:800]
        sc = StandardScaler(copy=False)
        Xtr = scale_reshape(windows, tr, sc, L, F, fit=True)
        Xte = scale_reshape(windows, te, sc, L, F, fit=False)
        yv = y.astype(np.int64) if task == "clf" else y.astype(np.float32)
        pred, ep = train_fold(Xtr, yv[tr], Xte, task, sample_ids[tr], device)
        if task == "clf":
            s = float(f1_score(yv[te], pred, average="macro", zero_division=0))
        else:
            s = float(mean_absolute_error(yv[te], pred))
        scores.append(s)
        p(tag, "FOLD", fold=fold, epochs=ep, score=round(s, 4))
        del Xtr, Xte
        if SMOKE:
            break
    scores = np.array(scores)
    name = "macroF1" if task == "clf" else "MAE"
    log(f"[{tag}] {target} cnn1d W={wl} {name} mean={scores.mean():.4f} std={scores.std():.4f} "
        f"per_fold={[round(float(x), 4) for x in scores]} device={device} ({time.time()-t_start:.0f}s)")
    p(tag, "DONE", total_s=round(time.time() - t_start))


if __name__ == "__main__":
    main()
