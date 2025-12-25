import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mrtf.config import Config
from mrtf.models.core import MRTFModel, fusion_regularization
from mrtf.rl.rasl import RASLModule
from mrtf.xai.explain import integrated_gradients, ecs

def bce(yhat, y):
    eps = 1e-7
    yhat = torch.clamp(yhat, eps, 1-eps)
    return -(y*torch.log(yhat) + (1-y)*torch.log(1-yhat)).mean()

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
        return {
            "voice": torch.from_numpy(self.voice[i]),
            "mri": torch.from_numpy(self.mri[i]),
            "sensor": torch.from_numpy(self.sensor[i]),
            "y": torch.tensor(self.y[i], dtype=torch.float32)
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = Config()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasl = RASLModule(init_threshold=0.5).to(device)
    model = MRTFModel(cfg, rasl).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ds = DummyDataset(200, cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for ep in range(cfg.epochs):
        model.train()
        accs, losses = [], []

        for b in dl:
            v, m, s, y = b["voice"].to(device), b["mri"].to(device), b["sensor"].to(device), b["y"].to(device)
            yhat, (Ev, Em, Es, H, _) = model(v, m, s)

            H_det = H.detach().requires_grad_(True)
            phi = integrated_gradients(lambda z: torch.sigmoid(model.clf.net(z).squeeze(-1)), H_det)
            ecs_val = ecs(phi).mean()

            thr = model.rasl.threshold()
            pred = (yhat >= thr).float()
            acc = (pred == y).float().mean()
            err = 1.0 - acc
            R = cfg.alpha1_acc*acc + cfg.alpha2_ecs*ecs_val - cfg.alpha3_err*err

            model.rasl.update_baseline(R.detach())
            L_RL = model.rasl.rl_loss(R.view(1))

            L = bce(yhat, y) + 0.1*fusion_regularization(Ev, Em, Es) + cfg.lambda_rl*L_RL

            opt.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            accs.append(acc.item())
            losses.append(L.item())

        print(f"epoch={ep+1} loss={np.mean(losses):.4f} acc={np.mean(accs):.4f} thr={model.rasl.threshold():.3f}")

if __name__ == "__main__":
    main()
