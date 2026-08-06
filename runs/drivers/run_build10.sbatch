#!/bin/bash
#SBATCH --job-name=build10
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/scratch/build10_logs/build10_%j.out
set -eo pipefail
SRC=/path/to/scratch/misoperation_assessment/scratch
B=/path/to/scratch/build10full
OUT=/path/to/datasets/PROTECT-90/windows/PROTECT-90
trap 'rm -rf "$B"' EXIT          # always reclaim the ~60GB unzip
rm -rf "$B"; mkdir -p "$B"
echo "UNZIP START $(date)"
unzip -q "$SRC/preprocessed_data.zip" -d "$B"
mv "$B/hv_double_line_90kv_preprocessed_data" "$B/preprocessed_data"
cp "$SRC/labels.csv" "$B/labels.csv"
echo "UNZIP DONE $(date); episodes=$(ls "$B/preprocessed_data" | wc -l)"; df -h "$B" | tail -1
cd /path/to/repos/protection-eval-framework
BUILD10="$B" OUT10="$OUT" .venv/bin/python /path/to/scratch/build_windows_10ms.py
echo "BUILD DONE $(date)"
ls -la "$OUT"/*W0p010* 2>/dev/null
