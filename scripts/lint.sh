#!/usr/bin/env bash
set -euo pipefail
# Runs the full static-check + format-check gate (mirrors `make lint`).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -d .venv ]; then
  echo "[!] no .venv found — run Task A first"
  exit 1
fi
. .venv/bin/activate
ruff check w2v_factory tests scripts
black --check w2v_factory tests scripts
echo "[✓] lint + format check clean"