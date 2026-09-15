#!/bin/bash
#SBATCH --job-name=csv2root-test
#SBATCH --partition=express
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/disk/gfs_atp/xlzd/alpine/analysis/csv2root-test-%j.out
#SBATCH --error=/disk/gfs_atp/xlzd/alpine/analysis/csv2root-test-%j.err

set -euo pipefail
umask 000

SOURCE="/disk/gfs_atp/xlzd/alpine/run1_2/20260814T122238Z-multi-ch-100c-0-5-2-6-54v-400hz-10ns-3-3v-d3b1fc20/waveforms-0000.csv.gz"

REFERENCE="/disk/gfs_atp/xlzd/alpine/run1_2/20260814T122238Z-multi-ch-100c-0-5-2-6-54v-400hz-10ns-3-3v-d3b1fc20/waveforms-0000.root"

SCRIPT="/disk/gfs_atp/xlzd/alpine/convert/csvgz_to_alpine_root.py"

OUT="/disk/gfs_atp/xlzd/alpine/analysis/reconstructed-test.root"

: "${PYTHON_EXE:?PYTHON_EXE was not exported to the Slurm job}"

SCRATCH="/scratch/$USER/$SLURM_JOB_ID"

cleanup() {
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

mkdir -p "$SCRATCH"

echo "HOST=$(hostname)"
echo "USER=$(whoami)"
echo "PYTHON_EXE=$PYTHON_EXE"
echo "SCRIPT=$SCRIPT"
echo

echo "=== Dependency check ==="
"$PYTHON_EXE" - <<'PY'
import sys
import numpy
import uproot
print("Python:", sys.executable)
print("NumPy :", numpy.__version__)
print("uproot:", uproot.__version__)
PY

echo
echo "=== Input ==="
ls -lh "$SOURCE"
ls -lh "$REFERENCE"
ls -lh "$SCRIPT"

echo
echo "=== Copy to scratch ==="
cp "$SOURCE" "$SCRATCH/input.csv.gz"
cp "$REFERENCE" "$SCRATCH/reference.root"

echo
echo "=== Convert ==="
"$PYTHON_EXE" "$SCRIPT" \
    "$SCRATCH/input.csv.gz" \
    "$SCRATCH/reconstructed.root" \
    --compare-reference "$SCRATCH/reference.root"

echo
echo "=== Result ==="
ls -lh "$SCRATCH/reconstructed.root"

echo
echo "=== Copy back ==="
cp "$SCRATCH/reconstructed.root" "$OUT"
chmod 666 "$OUT" || true

echo
echo "DONE: $OUT"
