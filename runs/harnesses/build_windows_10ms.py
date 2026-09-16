"""Build 10ms windows via psp_helper.create_windows (programmatic, pinned).

Drives create_windows with the framework config (bypassing @hydra.main via
__wrapped__), replicating the original build fingerprint
(step=0.005, topology=hv_double_line_90kv) at window_length=0.010.

Env:
    BUILD10 -> dir containing preprocessed_data/ + labels (dataset root)
    OUT10   -> output windows dir
"""
import os

from hydra import compose, initialize_config_dir

CFG_DIR = "/path/to/repos/protection-eval-framework/config"
BUILD = os.environ["BUILD10"]
OUT = os.environ["OUT10"]

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(
        config_name="main-config",
        overrides=[
            "window_extraction.window_length=0.010",
            "window_extraction.step_length_seconds=0.005",
            f"window_extraction.windows_local_dir={OUT}",
            f"dataset.tmp_data_directory={BUILD}",
            f"dataset.dataset_directory={BUILD}",
            f"dataset.data_directory={BUILD}",
            "dataset.topology=hv_double_line_90kv",
        ],
    )

from omegaconf import OmegaConf  # noqa: E402

# Published PROTECT-90 labels are comma-separated and indexed by sample_id
# (the framework config defaults target a different label export).
OmegaConf.set_struct(cfg, False)
cfg.dataset.index_col = "sample_id"
cfg.dataset.seperator = ","
if "raw_foldername" not in cfg.dataset:
    cfg.dataset.raw_foldername = ""  # only used by the optional QA report

import glob  # noqa: E402

from psp_helper.create_windows import create_windows  # noqa: E402

try:
    create_windows.__wrapped__(cfg)
except Exception as e:  # the optional post-build QA report needs extra keys; ignore
    print("create_windows raised after window write (likely optional report):", repr(e), flush=True)

# Success is defined by the combined windows + atomic done-marker, written BEFORE the report.
xs = glob.glob(os.path.join(OUT, "X_*W0p010*.raw"))
done = glob.glob(os.path.join(OUT, "X_*W0p010*.manifest.done"))
if xs and done:
    print("BUILD10_DONE ->", xs, flush=True)
else:
    raise SystemExit(f"BUILD10_FAILED: no 10ms window output in {OUT} (xs={xs}, done={done})")
