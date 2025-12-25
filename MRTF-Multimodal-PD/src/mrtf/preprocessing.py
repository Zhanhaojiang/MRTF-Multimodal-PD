import torch

def preprocess_voice(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.std(dim=(-1, -2), keepdim=True) + 1e-6)

def preprocess_mri(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=(-1, -2), keepdim=True)) / (x.std(dim=(-1, -2), keepdim=True) + 1e-6)

def preprocess_sensor(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
