"""Correlated multi-asset Heston simulator and Black-Scholes helpers (all in torch).

Each underlying follows Heston stochastic volatility, discretised with full-truncation
Euler. The underlyings' price shocks are correlated with each other, and each price
shock is correlated with its own variance shock (the leverage effect).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HestonParams:
    s0: float = 100.0
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.5
    rho: float = -0.7        # price/variance correlation
    asset_corr: float = 0.5  # correlation between underlyings' price shocks
    n_assets: int = 2
    steps_per_year: int = 252


def simulate_heston(p: HestonParams, n_paths: int, n_steps: int,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (S, V), each of shape [n_paths, n_steps + 1, n_assets]."""
    dt = 1.0 / p.steps_per_year
    k = p.n_assets
    corr = torch.full((k, k), p.asset_corr)
    corr.fill_diagonal_(1.0)
    chol = torch.linalg.cholesky(corr)

    S = torch.empty(n_paths, n_steps + 1, k)
    V = torch.empty(n_paths, n_steps + 1, k)
    S[:, 0], V[:, 0] = p.s0, p.v0
    for t in range(n_steps):
        z_s = torch.randn(n_paths, k, generator=generator) @ chol.T
        z_perp = torch.randn(n_paths, k, generator=generator)
        z_v = p.rho * z_s + math.sqrt(1 - p.rho**2) * z_perp
        v = V[:, t].clamp(min=0.0)
        sv = torch.sqrt(v * dt)
        S[:, t + 1] = S[:, t] * torch.exp(-0.5 * v * dt + sv * z_s)
        V[:, t + 1] = V[:, t] + p.kappa * (p.theta - v) * dt + p.xi * sv * z_v
    return S, V.clamp(min=0.0)


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2.0)))


def bs_call(s: torch.Tensor, k: float, tau: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    """Black-Scholes call price with zero rates; tau in years. At tau = 0 returns payoff."""
    tau = torch.as_tensor(tau, dtype=s.dtype)
    payoff = (s - k).clamp(min=0.0)
    vol = vol.clamp(min=1e-4)
    sq = vol * torch.sqrt(tau.clamp(min=1e-12))
    d1 = (torch.log(s / k) + 0.5 * sq**2) / sq
    d2 = d1 - sq
    price = s * _norm_cdf(d1) - k * _norm_cdf(d2)
    return torch.where(tau > 0, price, payoff)


def bs_delta(s: torch.Tensor, k: float, tau: torch.Tensor, vol: torch.Tensor) -> torch.Tensor:
    tau = torch.as_tensor(tau, dtype=s.dtype)
    vol = vol.clamp(min=1e-4)
    sq = vol * torch.sqrt(tau.clamp(min=1e-12))
    d1 = (torch.log(s / k) + 0.5 * sq**2) / sq
    return torch.where(tau > 0, _norm_cdf(d1), (s > k).to(s.dtype))
