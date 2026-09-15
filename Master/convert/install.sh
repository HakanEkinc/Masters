#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/alpine-convert}"

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --upgrade "$HERE"

echo
echo "Installed:"
alpine-convert --version
echo
echo "Activate later with:"
echo "  source $VENV_DIR/bin/activate"
