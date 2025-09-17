#!/usr/bin/env bash
set -euo pipefail
# Simplest form: assumes configs/ablation_grid.yaml and its base_cfg already
# point to a config where DATA.input_files is set properly.
# Usage:
#   conda activate w2v-factory
#   bash scripts/run_ablation.sh
python -m w2v_factory.cli.ablate --grid configs/ablation_grid.yaml
