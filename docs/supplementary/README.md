# Supplementary Material (Code)

This repository is intended to accompany the associated paper as supplementary code.

## What is included
- A runnable reference implementation of:
  - modality encoders (Voice/MRI/Sensor),
  - CAFT fusion,
  - XAI utilities (attribution, counterfactuals, ECS),
  - RASL-style reinforcement-assisted self-learning,
  - dummy/synthetic data so the pipeline runs end-to-end.
- Reproducibility helpers (global seeding, pinned requirements).

## What is NOT included
- Proprietary or restricted datasets.
- Full preprocessing pipelines that require dataset-specific details (paths, licenses, clinical metadata).
  Instead, we provide scaffolding and clear integration points.

## How to reproduce the demo outputs
```bash
bash scripts/reproduce_supplementary.sh
```

Outputs are written to:
- `results/demo_metrics.json`
- `results/demo_eval.json`

## Integrating your dataset
Use `src/mrtf/data/datasets.py` as a template and generate an `index.csv` with the required columns.
