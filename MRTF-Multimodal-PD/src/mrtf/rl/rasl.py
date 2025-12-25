import math
import torch
import torch.nn as nn

class RASLModule(nn.Module):
    def __init__(self, init_threshold=0.5, beta=0.9):
        super().__init__()
        self.threshold_logit = nn.Parameter(torch.tensor([math.log(init_threshold/(1-init_threshold))], dtype=torch.float32))
        self.register_buffer("reward_baseline", torch.tensor(0.0))
        self.beta = beta

    def threshold(self) -> float:
        return torch.sigmoid(self.threshold_logit).item()

    @torch.no_grad()
    def update_baseline(self, r: torch.Tensor):
        r_mean = r.mean().detach()
        self.reward_baseline = self.beta * self.reward_baseline + (1 - self.beta) * r_mean

    def rl_loss(self, rewards: torch.Tensor) -> torch.Tensor:
        adv = rewards - self.reward_baseline
        return -(adv.mean())
