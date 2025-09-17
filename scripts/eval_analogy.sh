#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   conda activate w2v-factory
#   bash scripts/eval_analogy.sh /path/to/embeddings.txt [/path/to/questions-words.txt]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VEC="${1:-$ROOT/runs/exp1/embeddings.txt}"
ANA="${2:-$ROOT/questions-words.txt}"
python -m w2v_factory.cli.eval --vectors "$VEC" --analogy "$ANA"
