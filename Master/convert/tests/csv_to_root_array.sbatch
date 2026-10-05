#!/bin/bash
#SBATCH --job-name=csv2root
#SBATCH --partition=express
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/disk/gfs_atp/xlzd/alpine/analysis/csv2root-logs/slurm-%A_%a.out
#SBATCH --error=/disk/gfs_atp/xlzd/alpine/analysis/csv2root-logs/slurm-%A_%a.err

set -euo pipefail
umask 000

SCRIPT="/disk/gfs_atp/xlzd/alpine/convert/csvgz_to_alpine_root.py"

: "${MANIFEST:?MANIFEST not supplied}"
: "${PYTHON_EXE:?PYTHON_EXE not supplied}"

LINE=$((SLURM_ARRAY_TASK_ID + 1))
SOURCE=$(sed -n "${LINE}p" "$MANIFEST")

if [[ -z "$SOURCE" ]]; then
    echo "No manifest entry for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

DEST="${SOURCE%.csv.gz}.root"

echo "HOST       = $(hostname)"
echo "TASK       = $SLURM_ARRAY_TASK_ID"
echo "SOURCE     = $SOURCE"
echo "DEST       = $DEST"
echo "PYTHON     = $PYTHON_EXE"

# Safety: never overwrite an existing ROOT file.
if [[ -f "$DEST" ]]; then
    echo "[SKIP] ROOT already exists: $DEST"
    exit 0
fi

SCRATCH="/scratch/$USER/csv2root/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"

SUCCESS=0

cleanup() {
    if [[ "$SUCCESS" -eq 1 ]]; then
        rm -rf "$SCRATCH"
    else
        echo "[NOTE] Failure: scratch retained at $SCRATCH" >&2
    fi
}
trap cleanup EXIT

echo "Copying CSV.GZ to scratch..."
cp "$SOURCE" "$SCRATCH/input.csv.gz"

echo "Converting..."
"$PYTHON_EXE" "$SCRIPT" \
    "$SCRATCH/input.csv.gz" \
    "$SCRATCH/output.root"

echo "Copying ROOT back beside CSV..."
TMPDEST="${DEST}.copying.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cp "$SCRATCH/output.root" "$TMPDEST"
chmod 666 "$TMPDEST" || true

# Rename only once the complete ROOT has arrived.
mv "$TMPDEST" "$DEST"
chmod 666 "$DEST" || true

SUCCESS=1

echo "[DONE] $DEST"
