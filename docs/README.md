# Documentation

- See `notebooks/MRTF_implementation.ipynb` for an end-to-end runnable walkthrough.
- Core modules:
  - `src/mrtf/models/core.py` — encoders, CAFT, classifier
  - `src/mrtf/xai/explain.py` — IG attribution, counterfactuals, ECS
  - `src/mrtf/rl/rasl.py` — RASL threshold + reward baseline
