"""Pre-trained transformer baseline (MOMENT-1-large) through the IDENTICAL evaluation pipeline.

Adds a LARGE, pre-trained time-series foundation model alongside the simple models and the
from-scratch 1D-CNN, to test whether foundation-model embeddings transfer to power-system
protection signals. The MOMENT backbone is FROZEN; each window is encoded to a fixed-dim
embedding and two comparators are reported per cell:

  * probe : a LINEAR probe on the frozen embeddings (LogisticRegression / regularized Ridge) -
            the "do the learned embeddings transfer?" head-to-head vs the simple models.
  * head  : a small MLP head on the same frozen embeddings (stronger, competitive).

Channels: MOMENT is channel-independent. Rather than average the 48 relay channels into one
vector (which discards the spatial signature that carries fault type / location), we take the
PER-CHANNEL embedding (reduction="none" -> mask-pool over valid patches per channel) and
concatenate -> a fixed 48*d_model feature per window. Stored as fp16 to fit memory.

FOLD-LOCAL PROTOCOL (one SLURM job == one outer fold): for the selected outer fold the raw-window
StandardScaler is fit ONLY on that fold's training episodes, then applied unchanged to the fold's
training and test windows. The frozen MOMENT encoder is applied separately within the fold to
produce fold-specific train and test embeddings (no global embedding matrix is shared across
folds). Probe preprocessing, probe fitting, and the episode-grouped validation split used for
MLP-head early stopping are restricted to the outer-training partition; the held-out fold is used
only for final prediction and scoring. Episode-grouped 5-fold GroupKFold, onset-only FL validity,
macro-F1 (FC) / MAE %line (FL), aggregated as mean +/- std(ddof=0) over the 5 folds by a separate
aggregation script.

Each job writes one atomic JSON to MOMENT_PART_OUT. Run in the py_tsfm conda env
(momentfm/transformers/huggingface):
    HF_HOME=/path/to/hf_cache HF_HUB_OFFLINE=1 MOMENT_PART_OUT=<path> \
    /path/to/conda/envs/py_tsfm/bin/python \
        runs/harnesses/moment_baseline.py <tag> <target> <wl> <outer_fold>
"""
import json
import os
import platform
import sys
import time

import numpy as np
import sklearn
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import momentfm
from momentfm import MOMENTPipeline

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")
WINDIR = os.environ.get("WINDIR", "/path/to/datasets/PROTECT-90/windows/PROTECT-90")
PART_OUT = os.environ.get("MOMENT_PART_OUT", "")       # REQUIRED: atomic per-fold JSON output path
SEED = int(os.environ.get("MOMENT_SEED", "42"))
SMOKE = os.environ.get("MOMENT_SMOKE", "0") == "1"     # plumbing-only: one real fold on a reduced
                                                       # window subset, 3 head epochs (metric NOT meaningful)
SMOKE_N = int(os.environ.get("MOMENT_SMOKE_N", "4000"))
MODEL_ID = os.environ.get("MOMENT_MODEL", "AutonLab/MOMENT-1-large")
# Pin the exact Hugging Face commit so the frozen weights are reproducible (the bare repo id can move).
MODEL_REVISION = os.environ.get("MOMENT_MODEL_REVISION", "ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc")
EPOCHS = int(os.environ.get("MOMENT_EPOCHS", "200"))   # max MLP-head epochs
PATIENCE = int(os.environ.get("MOMENT_PATIENCE", "20"))  # early-stop patience (val loss)
PROBE_CAP = int(os.environ.get("MOMENT_PROBE_CAP", "40000"))  # cap linear-probe FIT rows (memory; FC has 209k)
BSZ = int(os.environ.get("MOMENT_BSZ", "32"))          # embedding-extraction batch (windows)
SCALER_CHUNK = int(os.environ.get("MOMENT_SCALER_CHUNK", "4096"))  # chunked partial_fit batch
# MLP-head hyperparameters (fixed a priori; recorded in every per-fold JSON for auditability).
HEAD_LR = float(os.environ.get("MOMENT_HEAD_LR", "1e-3"))
HEAD_WD = float(os.environ.get("MOMENT_HEAD_WD", "1e-4"))
HEAD_BATCH = int(os.environ.get("MOMENT_HEAD_BATCH", "256"))
VAL_FRAC = float(os.environ.get("MOMENT_VAL_FRAC", "0.1"))   # episode-grouped inner-validation fraction
HEAD_DROPOUT = float(os.environ.get("MOMENT_HEAD_DROPOUT", "0.2"))
HEAD_HIDDEN = (512, 256)
MOMENT_LEN = 512                                        # MOMENT fixed input length
PATCH_LEN = 8                                           # MOMENT patch length -> 64 patches
N_PATCH = MOMENT_LEN // PATCH_LEN


def p(tag, stage, **kv):
    print(("[%s] %s %s" % (tag, stage, " ".join("%s=%s" % kv_ for kv_ in kv.items()))).rstrip(), flush=True)


def _pkgver(name):
    """Best-effort installed-package version (recorded in each per-fold JSON environment block)."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def embed_indices(model, windows, indices, scaler, device, d_model):
    """Frozen MOMENT PER-CHANNEL embedding for the given window indices only.

    (indices,) select rows of ``windows`` (N, L, F) -> apply the fold-training-fitted per-feature
    StandardScaler on the flattened window -> per batch (b, F, L) -> pad to (b, F, 512)+mask ->
    reduction='none' (b, F, P, d_model) -> mask-pool valid patches -> (b, F, d_model) -> flatten
    (b, F*d_model), fp16. The scaler is fit only on this fold's training episodes; the same scaler
    transforms both the training and test indices, so the embeddings are fold-specific."""
    indices = np.asarray(indices)
    n = len(indices)
    _, L, F = windows.shape
    out = np.empty((n, F * d_model), dtype=np.float16)
    use_amp = device == "cuda"
    for i in range(0, n, BSZ):
        idx = indices[i:i + BSZ]
        raw = np.asarray(windows[idx], dtype=np.float32).reshape(len(idx), L * F)
        xb = scaler.transform(raw).reshape(-1, L, F).transpose(0, 2, 1)   # (b, F, L), channel-first, scaled
        b = xb.shape[0]
        x = np.zeros((b, F, MOMENT_LEN), dtype=np.float32)
        x[:, :, :L] = xb
        mask = np.zeros((b, MOMENT_LEN), dtype=np.float32)
        mask[:, :L] = 1.0
        xt = torch.from_numpy(x).to(device)
        mt = torch.from_numpy(mask).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            e = model.embed(x_enc=xt, input_mask=mt, reduction="none").embeddings   # (b, F, P, d_model)
        e = e.float()
        # valid-patch mask (a patch is valid iff any of its PATCH_LEN samples is valid)
        mp = mt.reshape(b, N_PATCH, PATCH_LEN).amax(dim=2)                           # (b, P)
        w = mp[:, None, :, None]                                                     # (b, 1, P, 1)
        pooled = (e * w).sum(dim=2) / w.sum(dim=2).clamp(min=1.0)                     # (b, F, d_model)
        out[i:i + b] = pooled.reshape(b, F * d_model).half().cpu().numpy()
        if (i // BSZ) % 200 == 0:
            print("[embed] %d/%d" % (i, n), flush=True)
    return out


def make_head(d_in, n_out):
    # Fixed baseline head configuration. No architecture or hyperparameter selection is performed
    # using the corrected outer-test folds. A leading BatchNorm standardizes the features online
    # (scales to FC's 209k via minibatch, no 41 GB standardized matrix); dropout curbs overfitting.
    h1, h2 = HEAD_HIDDEN
    return nn.Sequential(
        nn.BatchNorm1d(d_in),
        nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(HEAD_DROPOUT),
        nn.Linear(h2, n_out))


def train_mlp_head(Etr, ytr, Ete, task, sids_tr, device, generator, tag=""):
    """Train the MLP head on the outer-training embeddings; episode-grouped early stopping on val
    loss, best-val checkpoint used for prediction. The GroupShuffleSplit validation fold is drawn
    only from the outer-training partition, so the held-out outer fold never enters early stopping,
    loss monitoring, BatchNorm fitting, or checkpoint selection."""
    from sklearn.model_selection import GroupShuffleSplit
    n_out = (int(ytr.max()) + 1) if task == "clf" else 1
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    tr_i, va_i = next(gss.split(Etr, ytr, groups=sids_tr))
    yt = torch.tensor(ytr, dtype=(torch.long if task == "clf" else torch.float32))
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Etr[tr_i]), yt[tr_i]),
                       batch_size=HEAD_BATCH, shuffle=True, generator=generator)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Etr[va_i]), yt[va_i]), batch_size=512, shuffle=False)
    model = make_head(Etr.shape[1], n_out).to(device)
    n_params = int(sum(pp.numel() for pp in model.parameters() if pp.requires_grad))
    opt = torch.optim.Adam(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    lossf = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()
    max_ep = 3 if SMOKE else EPOCHS
    best_val, best_state, wait, stopped = np.inf, None, 0, False
    epoch = 0
    for epoch in range(max_ep):
        model.train(); tloss = 0.0; ntb = 0
        for xb, yb in dl_tr:
            xb, yb = xb.float().to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb) if task == "clf" else lossf(out.squeeze(1), yb)
            loss.backward(); opt.step()
            tloss += loss.item(); ntb += 1
        model.eval(); vloss = 0.0; nb = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.float().to(device), yb.to(device)
                out = model(xb)
                vloss += (lossf(out, yb) if task == "clf" else lossf(out.squeeze(1), yb)).item(); nb += 1
        vloss /= max(nb, 1); tloss /= max(ntb, 1)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == max_ep - 1:
            p(tag, "TRAJ", kind="head", epoch=epoch + 1, train_loss=round(tloss, 5), val_loss=round(vloss, 5))
        if vloss < best_val - 1e-5:
            best_val, best_state, wait = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                stopped = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Ete), 512):
            out = model(torch.from_numpy(Ete[i:i + 512]).float().to(device))
            preds.append((out.argmax(1) if task == "clf" else out.squeeze(1)).cpu().numpy())
    return np.concatenate(preds), epoch + 1, stopped, n_params


def _metric(task, ytrue, pred):
    if task == "clf":
        return float(f1_score(ytrue, pred, average="macro", zero_division=0))
    return float(mean_absolute_error(ytrue, pred))


def _atomic_write_json(path, payload):
    if not path:
        raise SystemExit("MOMENT_PART_OUT is required (per-fold JSON output path)")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def main():
    if len(sys.argv) < 5:
        raise SystemExit("usage: moment_baseline.py <tag> <target> <window_length> <outer_fold>")
    tag, target, wl = sys.argv[1:4]
    outer_fold = int(sys.argv[4])
    if not 1 <= outer_fold <= 5:
        raise SystemExit("outer_fold must be an integer in 1..5")
    t_start = time.time()

    # Explicit per-job random state (do NOT claim bitwise determinism across GPU models).
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(SEED)

    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="main-config", overrides=[
            f"training.target_label={target}",
            f"window_extraction.window_length={wl}",
            f"window_extraction.windows_local_dir={WINDIR}",
            "tracking.mode=disabled"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    model = MOMENTPipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, model_kwargs={"task_name": "embedding"})
    model.init()
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    d_model = int(model.config.d_model)
    p(tag, "RUN", target=target, wl=wl, outer_fold=outer_fold, device=device, gpu=gpu_name,
      model=MODEL_ID, d_model=d_model, d_in=48 * d_model, seed=SEED, mode=("smoke" if SMOKE else "full"))

    from fcl_psp.models.run_model import get_sample_ids_and_fault_targets
    from fcl_psp.models.moment_cv_utils import make_folds, fit_flat_window_scaler, probe_sklearn
    from psp_helper.windows_helper import load_windows_and_labels
    windows, labels, ri = load_windows_and_labels(cfg)
    # RACE-SAFE fault-only handling. Previously this wrote a filtered copy via
    # write_filtered_memmap(..., mode="w+") to a per-config path (X_fo_moment_{tag}.raw) shared by
    # ALL folds of that config; concurrent SLURM-array folds could truncate/rewrite it under one
    # another mid-embed, so a fold could read zeroed/partial rows (data race -> corrupted FL
    # embeddings). Fix: never write a filtered copy. `windows` is the ORIGINAL read-only memmap
    # (opened mode="r"); nothing writes it, so concurrent folds are safe. For FL, `ri` holds the
    # ORIGINAL window indices of the fault-only subset and the returned `labels` are already filtered
    # to that subset (aligned 1:1 with `ri`); for FC, `ri is None` and every window is used.
    base_indices = np.asarray(ri) if ri is not None else np.arange(windows.shape[0])
    sample_ids, y, labels = get_sample_ids_and_fault_targets(labels, cfg)
    sample_ids = np.asarray(sample_ids)
    y = np.asarray(y)
    task = "clf" if target == "event_type" else "reg"
    _, L, F = windows.shape
    N = int(base_indices.shape[0])       # number of samples the folds are built on (fault-only for FL)
    p(tag, "DATA", N=N, L=L, F=F, task=task)

    yv = y.astype(np.int64) if task == "clf" else y.astype(np.float32)
    # SMOKE restricts to a small window subset (plumbing only); the selected fold and the
    # train-only protocol are unchanged.
    sel = np.arange(min(N, SMOKE_N)) if SMOKE else np.arange(N)
    y_sel, sid_sel = yv[sel], sample_ids[sel]
    base_sel = base_indices[sel]         # ORIGINAL window indices for the selected (filtered) rows
    n_splits = int(cfg.training.n_splits)
    folds = make_folds(y_sel, sid_sel, n_splits=n_splits)
    if outer_fold > len(folds):
        raise SystemExit(f"outer_fold {outer_fold} exceeds n_splits {len(folds)}")
    tr_local, te_local = folds[outer_fold - 1]

    train_idx = base_sel[tr_local]       # ORIGINAL window indices into the read-only memmap
    test_idx = base_sel[te_local]
    y_train, y_test = y_sel[tr_local], y_sel[te_local]
    sids_train, sids_test = sid_sel[tr_local], sid_sel[te_local]

    train_groups = set(sids_train.tolist())
    test_groups = set(sids_test.tolist())
    group_overlap = len(train_groups & test_groups)
    assert train_groups.isdisjoint(test_groups), f"episode groups overlap: {group_overlap}"
    p(tag, "FOLD", outer_fold=outer_fold, n_train_windows=len(train_idx), n_test_windows=len(test_idx),
      n_train_groups=len(train_groups), n_test_groups=len(test_groups), group_overlap=group_overlap)

    # Raw-window StandardScaler fit ONLY on this fold's training windows (chunked partial_fit).
    raw_scaler = fit_flat_window_scaler(windows, train_idx, chunk_size=SCALER_CHUNK)
    assert int(raw_scaler.n_samples_seen_) == len(train_idx), \
        f"scaler saw {int(raw_scaler.n_samples_seen_)} windows, expected {len(train_idx)}"
    print("SCALER", flush=True)
    print(f"fold={outer_fold}", flush=True)
    print(f"fit_windows={int(raw_scaler.n_samples_seen_)}", flush=True)
    print(f"features={L * F}", flush=True)
    print("test_windows_seen=0", flush=True)

    te0 = time.time()
    E_train = embed_indices(model, windows, train_idx, raw_scaler, device, d_model)
    E_test = embed_indices(model, windows, test_idx, raw_scaler, device, d_model)
    emb_s = round(time.time() - te0)
    p(tag, "EMBED", n_train=len(E_train), n_test=len(E_test), d_in=E_train.shape[1], embed_s=emb_s)

    # Linear probe: scaler + estimator fit only on <=PROBE_CAP outer-training embeddings.
    pred_p, n_probe_fit = probe_sklearn(E_train, y_train, E_test, task, SEED, cap=PROBE_CAP)
    s_p = _metric(task, y_test, pred_p)
    p(tag, "PROBE", probe_fit_windows=n_probe_fit, probe_score=round(s_p, 4))

    # MLP head: grouped inner validation restricted to outer-training partition.
    pred_h, ep_h, st_h, head_params = train_mlp_head(E_train, y_train, E_test, task, sids_train, device, dl_generator, tag)
    s_h = _metric(task, y_test, pred_h)
    p(tag, "FOLD", outer_fold=outer_fold, probe=round(s_p, 4), probe_fit=n_probe_fit,
      head=round(s_h, 4), head_ep=ep_h, head_conv=st_h, head_params=head_params)

    del E_train, E_test

    # Out-of-fold predictions for independent metric recomputation / diagnostics: one row per held-out
    # test window (original window index, episode id, target, linear-probe and MLP-head predictions).
    preds_path = (PART_OUT[:-5] if PART_OUT.endswith(".json") else PART_OUT) + "_preds.npz"
    np.savez_compressed(
        preds_path,
        orig_window_index=np.asarray(test_idx),
        episode_id=np.asarray(sids_test),
        target=np.asarray(y_test),
        probe_pred=np.asarray(pred_p),
        head_pred=np.asarray(pred_h),
        outer_fold=np.asarray(outer_fold),
    )

    payload = {
        "tag": tag,
        "target": target,
        "task": task,
        "window_length": float(wl),
        "outer_fold": outer_fold,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "n_train_windows": int(len(train_idx)),
        "n_test_windows": int(len(test_idx)),
        "n_train_groups": int(len(train_groups)),
        "n_test_groups": int(len(test_groups)),
        "group_overlap": int(group_overlap),
        "raw_scaler_fit_windows": int(raw_scaler.n_samples_seen_),
        "probe_fit_windows": int(n_probe_fit),
        "probe_score": float(s_p),
        "head_score": float(s_h),
        "head_epochs": int(ep_h),
        "head_early_stopped": bool(st_h),
        "head_trainable_params": int(head_params),
        "hyperparameters": {
            "embed_batch": BSZ,
            "probe_cap": PROBE_CAP,
            "scaler_chunk": SCALER_CHUNK,
            "head_max_epochs": EPOCHS,
            "head_patience": PATIENCE,
            "head_val_frac": VAL_FRAC,
            "head_lr": HEAD_LR,
            "head_weight_decay": HEAD_WD,
            "head_batch": HEAD_BATCH,
            "head_dropout": HEAD_DROPOUT,
            "head_hidden": list(HEAD_HIDDEN),
            "head_batchnorm": True,
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "cuda_version": (torch.version.cuda or "cpu"),
        "transformers_version": _pkgver("transformers"),
        "momentfm_version": _pkgver("momentfm"),
        "gpu": gpu_name,
        "runtime_seconds": round(time.time() - t_start),
        "smoke": bool(SMOKE),
        "predictions_file": os.path.basename(preds_path),
    }
    _atomic_write_json(PART_OUT, payload)
    p(tag, "DONE", outer_fold=outer_fold, out=PART_OUT, total_s=round(time.time() - t_start))


if __name__ == "__main__":
    main()
