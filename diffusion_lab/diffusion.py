"""Gaussian diffusion (DDPM, Ho et al. 2020) with an optional DDIM sampler (Song et al. 2020).

Forward process (fixed, no learning):
    q(x_t | x_0) = N( sqrt(ᾱ_t) x_0 , (1 − ᾱ_t) I )          ᾱ_t = ∏_{s≤t} (1 − β_s)

Training objective (simple ε-prediction loss):
    L = E_{x_0, t, ε} || ε − ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1−ᾱ_t) ε , t ) ||²

Reverse process: start from x_T ~ N(0, I) and iterate t = T−1 … 0, each step using
ε_θ to estimate the mean of p(x_{t−1} | x_t).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(name: str, timesteps: int) -> torch.Tensor:
    if name == "linear":
        # Ho et al. values, scaled so they stay sensible for T != 1000.
        scale = 1000 / timesteps
        betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=torch.float64)
        return betas.clamp(max=0.999)  # only matters for T < ~20
    if name == "cosine":
        # Nichol & Dhariwal 2021. Works better for small images.
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alphas_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown schedule {name!r}")


def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Gathers a[t] for a batch of timesteps and reshapes to broadcast over (b, c, h, w)."""
    return a.gather(0, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000, schedule: str = "linear"):
        super().__init__()
        self.timesteps = timesteps
        betas = make_beta_schedule(schedule, timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        reg = lambda name, x: self.register_buffer(name, x.float(), persistent=False)  # noqa: E731
        reg("betas", betas)
        reg("alphas_bar", alphas_bar)
        reg("sqrt_alphas_bar", alphas_bar.sqrt())
        reg("sqrt_one_minus_alphas_bar", (1 - alphas_bar).sqrt())
        # Coefficients of the posterior q(x_{t-1} | x_t, x_0), eq. 6-7 of the DDPM paper.
        reg("posterior_variance", betas * (1 - alphas_bar_prev) / (1 - alphas_bar))
        reg("posterior_mean_coef1", betas * alphas_bar_prev.sqrt() / (1 - alphas_bar))
        reg("posterior_mean_coef2", (1 - alphas_bar_prev) * alphas.sqrt() / (1 - alphas_bar))

    # ---- forward process -------------------------------------------------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) in closed form."""
        return (_extract(self.sqrt_alphas_bar, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_alphas_bar, t, x0.shape) * noise)

    def training_loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """The simple ε-prediction MSE, with t sampled uniformly per example."""
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    # ---- reverse process -------------------------------------------------------------

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Invert q_sample: x_0 = (x_t − sqrt(1−ᾱ_t) ε) / sqrt(ᾱ_t)."""
        return ((x_t - _extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps)
                / _extract(self.sqrt_alphas_bar, t, x_t.shape))

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                 clip_x0: bool = True) -> torch.Tensor:
        """One ancestral DDPM step: x_{t-1} ~ p_θ(x_{t-1} | x_t)."""
        eps = model(x_t, t)
        x0 = self.predict_x0(x_t, t, eps)
        if clip_x0:
            x0 = x0.clamp(-1, 1)
        mean = (_extract(self.posterior_mean_coef1, t, x_t.shape) * x0
                + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = _extract(self.posterior_variance, t, x_t.shape)
        nonzero = (t > 0).float().reshape(-1, *([1] * (x_t.dim() - 1)))  # no noise at t = 0
        return mean + nonzero * var.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...], device=None,
               progress: bool = False) -> torch.Tensor:
        """Full DDPM ancestral sampling (T model evaluations)."""
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device)
        steps = range(self.timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            steps = tqdm(steps, desc="ddpm", leave=False)
        for i in steps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, shape: tuple[int, ...], steps: int = 50,
                    eta: float = 0.0, device=None, progress: bool = False) -> torch.Tensor:
        """DDIM sampling with `steps` evaluations. eta=0 is deterministic; eta=1 ≈ DDPM."""
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps + 1, device=device).long()
        pairs = list(zip(times[:-1].tolist(), times[1:].tolist()))
        if progress:
            from tqdm import tqdm
            pairs = tqdm(pairs, desc="ddim", leave=False)
        for i, (t_cur, t_next) in enumerate(pairs):
            t = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = model(x, t)
            x0 = self.predict_x0(x, t, eps).clamp(-1, 1)
            if i == len(pairs) - 1:  # last step lands exactly on the x0 estimate
                x = x0
                break
            a_cur, a_next = self.alphas_bar[t_cur], self.alphas_bar[t_next]
            sigma = eta * ((1 - a_next) / (1 - a_cur) * (1 - a_cur / a_next)).sqrt()
            dir_xt = (1 - a_next - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_next.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
        return x
