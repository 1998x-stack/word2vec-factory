#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   conda activate w2v-factory
#   bash scripts/train_cbow_hs.sh [/path/to/corpus.txt]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_BASE="$ROOT/configs/cbow_hs.yaml"
CORPUS="${1:-$ROOT/sample_corpus.txt}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/cbow_hs_$(date +%Y%m%d_%H%M%S)}"
TMP_CFG="$(mktemp --suffix=.yaml)"

cat > "$TMP_CFG" <<YAML
INCLUDE: $CFG_BASE
DATA:
  input_files:
    - $CORPUS
RUN:
  out_dir: $OUT_DIR
YAML

echo "[+] Training CBOW + Hierarchical Softmax"
python -m w2v_factory.cli.train --cfg "$TMP_CFG"
echo "[✓] Vectors -> $OUT_DIR/embeddings.txt"
