# MRTF: Multimodal Reinforcement‑Assisted Transformer Framework (Parkinson’s Detection)

> **Supplementary Code Release**: This repository accompanies the MRTF paper and provides a runnable reference implementation.
> It includes a dummy dataset for end-to-end execution and clear hooks for integrating real Voice/MRI/Sensor data.


This repository provides a **professional, runnable reference implementation** of the MRTF pipeline described in the provided diagram/method:  
**multimodal feature encoders → cross‑attention fusion (CAFT) → classifier → explainability → reinforcement‑assisted self‑learning (RASL)**.

> Note: The notebook and scripts include a **dummy/synthetic dataset** so the full pipeline can run end‑to‑end without proprietary data.  
> Replace the dummy dataset and preprocessing stubs with your real **Voice/MRI/Sensor** pipelines.

---

## Key Components

### Modal Encoders
- **Voice Encoder**: CNN → BiLSTM (spectrogram input)
- **MRI Encoder**: Vision Transformer (ViT‑B/16) with a CNN fallback if ViT is unavailable
- **Sensor Encoder**: Temporal Convolutional Network (TCN) → Temporal Transformer

### Fusion: CAFT (Cross‑Attention Fusion Transformer)
Tri‑directional cross‑attention across modalities with learnable fusion weights, followed by a Transformer encoder to produce a fused representation.

### Explainability (XAI)
- **Attribution**: Integrated Gradients (lightweight fallback for SHAP‑style explanations)
- **Counterfactuals**: Gradient‑based minimal perturbations in fused feature space
- **ECS**: Explanation‑Consistency Score (thresholded feature relevance summary)

### Reinforcement‑Assisted Self‑Learning (RASL)
A lightweight RL-style module that maintains a learnable decision threshold and a moving reward baseline; training incorporates an RL loss term driven by a reward that can combine:
- predictive accuracy,
- ECS,
- error penalties.

---

## Repository Layout

```
MRTF-Multimodal-PD/
  notebooks/                 # runnable demo notebook(s)
  src/mrtf/                  # modular Python package
    models/                  # encoders + CAFT + classifier
    xai/                     # attribution, counterfactual, ECS
    rl/                      # RASL module
  scripts/                   # training/evaluation entrypoints
  data/raw/                  # ignored by git (place raw data here)
  data/processed/            # ignored by git (place processed tensors here)
  configs/                   # experiment configs (optional)
  docs/                      # extended documentation
  tests/                     # lightweight sanity tests
```

---

## Quickstart

### 1) Create and activate an environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 3) Run the demo notebook
Open:
- `notebooks/MRTF_implementation.ipynb`

### 4) Run dummy training from the CLI
```bash
python scripts/train_dummy.py --epochs 3 --batch_size 8
```

---

## Using Real Data

### Voice
Expected model input: **spectrogram** tensor shaped `[B, 1, F, T]`.

Replace `preprocess_voice` in `src/mrtf/preprocessing.py` with your pipeline, e.g.:
- pre‑emphasis,
- STFT / MFCC,
- normalization and augmentation.

### MRI
Expected model input: image tensor shaped `[B, 3, 224, 224]`.

For single‑channel MRI:
- replicate the channel to 3, or
- modify the MRI encoder to accept 1 channel.

Typical MRI preprocessing (outside the model):
- skull stripping,
- bias field correction,
- intensity normalization,
- registration/resampling to a standard space.

### Wearable Sensors
Expected model input: time‑series tensor shaped `[B, C, L]` (e.g., accel+gyro).

Replace `preprocess_sensor` with your pipeline, e.g.:
- windowing (e.g., 5 seconds),
- denoising (wavelet),
- frequency features (FFT),
- normalization.

---

## Reproducibility Notes
- The provided implementation favors clarity and modularity over maximum performance.
- For production training, you will likely want:
  - proper dataset splits and stratification,
  - calibration (temperature scaling),
  - robust evaluation (ROC‑AUC, PR‑AUC),
  - systematic hyperparameter search.

---

## Contributing
Contributions are welcome (bug fixes, docs, refactors, training utilities).  
Please open an issue or submit a PR.

---

## License
Released under the **MIT License**. See `LICENSE`.
