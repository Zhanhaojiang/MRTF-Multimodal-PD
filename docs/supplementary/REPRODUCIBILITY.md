# Reproducibility Checklist

This checklist is intended for supplementary material.

- [ ] Record exact Python version (e.g., 3.10.x)
- [ ] Record CUDA + GPU model (if applicable)
- [ ] Use `requirements-pinned.txt` or `environment.yml`
- [ ] Set seeds via `mrtf.repro.set_global_seed(seed)`
- [ ] Save configs, metrics, and checkpoints per run
- [ ] Report metrics with confidence intervals (bootstrap) for real datasets
