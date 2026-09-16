# No-op tabulate shim (same rationale as the wandb shim): fcl_psp.models.cv_reporting
# imports `from tabulate import tabulate` for human-readable reporting that the CNN
# baseline never triggers. Provides a callable `tabulate` that returns a plain string.
def tabulate(data, *args, **kwargs):
    try:
        return "\n".join("\t".join(map(str, row)) for row in data)
    except Exception:
        return str(data)


def __getattr__(name):
    def _noop(*args, **kwargs):
        return None
    return _noop
