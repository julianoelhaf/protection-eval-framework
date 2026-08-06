# No-op wandb shim. The CNN baseline runs in the py_dl conda env (torch + loaders)
# with tracking.mode=disabled, so wandb is imported by fcl_psp.models.run_model but
# never exercised. This stub satisfies the import without adding wandb to the env.
# Any attribute resolves to a no-op callable.
def __getattr__(name):
    def _noop(*args, **kwargs):
        return None
    return _noop
