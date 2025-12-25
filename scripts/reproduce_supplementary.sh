#!/usr/bin/env bash
set -euo pipefail

# Reproduce (demo) results used in supplementary material.
# This runs dummy training, evaluation, and exports a JSON summary to results/.

python scripts/train_dummy.py --epochs 3 --batch_size 8 --save_json results/demo_metrics.json
python scripts/eval_dummy.py  --batch_size 8 --load_json results/demo_metrics.json --out results/demo_eval.json

echo "Done. See results/ for outputs."
