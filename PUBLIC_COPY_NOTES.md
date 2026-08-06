# About this public copy

This repository is a scrubbed, history-free export of the authors' internal
working repository, prepared with `scripts/make_public_copy.sh`.

## Intentionally not included

- **Manuscript LaTeX sources** (`paper/*.tex` and build assets). The revision is
  under review; the sources will be added once the paper is published. The
  reproducibility generators in `paper/tables/` **are** included -- they rebuild
  every results table from the committed evidence in `reports/`.
- **Internal working notes** -- planning documents, review correspondence, and
  agent handoff notes, none of which are needed to run or audit the code.
- **Cluster run logs** (`runs/logs/`), which contain site-specific hostnames and
  absolute scratch paths. The aggregated results they produced are in `reports/`.
- **GitLab CI configuration**, which is specific to the authors' internal runner.

## Placeholders you must adapt

- Absolute paths appear as `/path/to/datasets`, `/path/to/scratch`, and `$HOME`.
  Point them at your own data root and scratch space (see `config/`).
- The companion package `psp_helper` (config schema, constants, windowing incl.
  the onset-validity rule) is a separate repository. Adjust the `psp_helper`
  requirement in `pyproject.toml` to wherever you obtain it.

## Data

The experiments run on the public PROTECT-90 dataset:
<https://doi.org/10.5281/zenodo.21109169>
