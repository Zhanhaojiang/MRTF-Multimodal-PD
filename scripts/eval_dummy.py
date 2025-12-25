import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mrtf.config import Config
from mrtf.models.core import MRTFModel
from mrtf.rl.rasl import RASLModule
from mrtf.eval import compute_binary_metrics, compute_roc_auc
from mrtf.repro import set_global_seed

class DummyDataset(Dataset):
    def __init__(self, n, cfg, seed=42):
        rng = np.random.default_rng(seed)
        self.y = rng.integers(0, 2, size=n).astype(np.int64)
        self.voice = rng.normal(0, 1, size=(n, 1, cfg.voice_freq_bins, cfg.voice_time_steps)).astype(np.float32)
        self.mri = rng.normal(0, 1, size=(n, 3, cfg.mri_size, cfg.mri_size)).astype(np.float32)
        self.sensor = rng.normal(0, 1, size=(n, cfg.sensor_channels, cfg.sensor_len)).astype(np.float32)
        pos = self.y == 1
        self.voice[pos] += 0.15
        self.mri[pos] += 0.05
        self.sensor[pos] += 0.10

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        import torch
        return {
            "voice": torch.from_numpy(self.voice[i]),
            "mri": torch.from_numpy(self.mri[i]),
            "sensor": torch.from_numpy(self.sensor[i]),
            "y": torch.tensor(self.y[i], dtype=torch.float32)
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="results/demo_eval.json")
    ap.add_argument("--load_json", type=str, default="")
    args = ap.parse_args()

    set_global_seed(args.seed, deterministic=True)

    cfg = Config()
    cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasl = RASLModule(init_threshold=0.5).to(device)
    model = MRTFModel(cfg, rasl).to(device)
    model.eval()

    ds = DummyDataset(200, cfg, seed=args.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    probs, ys = [], []
    with torch.no_grad():
        for b in dl:
            v, m, s, y = b["voice"].to(device), b["mri"].to(device), b["sensor"].to(device), b["y"].to(device)
            yhat, _ = model(v, m, s)
            probs.append(yhat.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(ys)

    thr = model.rasl.threshold()
    m = compute_binary_metrics(y_true, y_prob, threshold=thr)
    auc = compute_roc_auc(y_true, y_prob)

    payload = {
        "threshold": float(thr),
        "metrics": {
            "acc": m.acc,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "roc_auc": auc,
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
