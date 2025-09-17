#!/bin/sh
set -eu

ENV_NAME="${1:-w2v-factory}"
PY_VER="${2:-3.11}"

# Find conda base; this requires 'conda' on PATH
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "$CONDA_BASE" ]; then
  # Try common fallback
  if [ -d "$HOME/miniconda3" ]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [ -d "$HOME/anaconda3" ]; then
    CONDA_BASE="$HOME/anaconda3"
  else
    echo "[!] conda not found on PATH and no common install dir detected."
    echo "    Install Miniconda or ensure 'conda' is on PATH."
    exit 1
  fi
fi

# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[+] Creating conda env: $ENV_NAME (python=$PY_VER)"
conda create -y -n "$ENV_NAME" "python=$PY_VER"

echo "[+] Activating env"
# POSIX-compatible activation
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[+] Installing requirements"
pip install -r requirements.txt
pip install -e .

echo "[✓] Done. Activate with: conda activate $ENV_NAME"
