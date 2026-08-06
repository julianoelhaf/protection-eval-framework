#!/bin/bash
#SBATCH --job-name=compile_redline
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/repos/protection-eval-framework/runs/logs/build/compile_redline_%j.out
# Compile paper/ijepe_revision_redlined.tex (marked-up + clean) with tectonic (env `tex`).
# The conda `texlive` env is an incomplete install (no latex.ltx); tectonic is self-contained.
# TECTONIC_CACHE_DIR is pre-warmed on the login node so this runs without network. The 5 cosmetic
# \includesvg icons are shimmed (tectonic has no inkscape); they render normally on Overleaf.
set -eo pipefail
REPO=/path/to/repos/protection-eval-framework
export PATH=/path/to/conda/envs/tex/bin:$PATH
export TECTONIC_CACHE_DIR=/path/to/scratch/tectonic_cache
OUTROOT=/path/to/scratch/redline_compile/job_${SLURM_JOB_ID}
RESULTS=$REPO/reports/runs/redline_compile_results.txt
mkdir -p "$OUTROOT" "$REPO/runs/logs/build" "$(dirname "$RESULTS")"

echo "START compile_redline job=$SLURM_JOB_ID host=$(hostname) $(date)"
echo "tectonic: $(which tectonic)"; tectonic --version
echo "CACHE=$TECTONIC_CACHE_DIR   OUTROOT=$OUTROOT"

"$REPO/.venv/bin/python" "$REPO/runs/harnesses/compile_redline.py" \
  --src-dir "$REPO/paper" \
  --tex ijepe_revision_redlined.tex \
  --outroot "$OUTROOT" \
  --results "$RESULTS" \
  --cache "$TECTONIC_CACHE_DIR"

echo "--- PDFs produced ---"
ls -la "$OUTROOT"/*.pdf 2>/dev/null || echo "NO PDF PRODUCED"
echo "RESULTS=$RESULTS"
echo "DONE compile_redline job=$SLURM_JOB_ID $(date)"
