from typing import Optional
import torch

def integrated_gradients(model_fn, x: torch.Tensor, baseline: Optional[torch.Tensor]=None, steps: int = 16) -> torch.Tensor:
    x = x.detach()
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, steps, device=x.device).view(steps, 1, 1)
    x_interp = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    x_interp.requires_grad_(True)

    y = model_fn(x_interp.view(-1, x.shape[-1]))
    grads = torch.autograd.grad(y.sum(), x_interp, retain_graph=False, create_graph=False)[0]
    avg_grads = grads.mean(dim=0)
    return ((x - baseline) * avg_grads).detach()

def find_counterfactual(model_fn, x: torch.Tensor, target: float=0.5, steps: int=50, lr: float=0.1, l2_weight: float=0.01) -> torch.Tensor:
    x0 = x.detach()
    x_cf = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([x_cf], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            y = model_fn(x_cf)
            loss = (y - target).pow(2).mean() + l2_weight * (x_cf - x0).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return x_cf.detach()

def ecs(phi: torch.Tensor, tau: float=0.01, clinical_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    if clinical_mask is None:
        clinical_mask = torch.ones_like(phi)
    important = (phi.abs() > tau).float()
    return (important * clinical_mask).mean(dim=-1)
