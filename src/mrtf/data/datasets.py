"""Dataset I/O scaffolding for supplementary material.

This module intentionally avoids bundling proprietary datasets.
Users should adapt the loaders to their data layout and licensing constraints.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SamplePaths:
    voice_path: str
    mri_path: str
    sensor_path: str
    label: int

class MultimodalFolderDataset(Dataset):
    """Example dataset wrapper.

Expected:
- `index.csv` with columns: voice_path,mri_path,sensor_path,label
- paths are relative to `root`.

Data formats:
- voice: precomputed spectrogram .npy (F,T) or tensor saved via torch
- mri: image tensor .npy (C,H,W) or torch
- sensor: timeseries .npy (C,L) or torch
"""

    def __init__(self, root: str, index_csv: str):
        import csv
        self.root = root
        self.items = []
        with open(index_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append(SamplePaths(
                    voice_path=row["voice_path"],
                    mri_path=row["mri_path"],
                    sensor_path=row["sensor_path"],
                    label=int(row["label"]),
                ))

    def __len__(self) -> int:
        return len(self.items)

    def _load_any(self, p: str) -> torch.Tensor:
        full = os.path.join(self.root, p)
        if full.endswith(".pt") or full.endswith(".pth"):
            return torch.load(full, map_location="cpu")
        if full.endswith(".npy"):
            return torch.from_numpy(np.load(full)).float()
        raise ValueError(f"Unsupported file format: {full}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        it = self.items[idx]
        voice = self._load_any(it.voice_path)
        mri = self._load_any(it.mri_path)
        sensor = self._load_any(it.sensor_path)
        y = torch.tensor(it.label, dtype=torch.float32)
        # Ensure expected shapes
        if voice.ndim == 2:
            voice = voice.unsqueeze(0)  # [1,F,T]
        return {"voice": voice, "mri": mri, "sensor": sensor, "y": y}
