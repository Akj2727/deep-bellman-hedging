"""Batched hedging environment.

The book is short one at-the-money call on each of `n_assets` Heston underlyings and
hedges by holding the underlyings. Each step the agent asks for target holdings; the
request then passes through a constraint layer (the same one every strategy uses):

  * trade limit:  |new - old| <= max_trade per asset per step (liquidity)
  * position box: 0 <= holding <= max_hold
  * budget:       sum(holding * price) <= budget * sum(initial prices); scaled down if over

Executed trades pay a proportional cost plus a quadratic market-impact cost.
Wealth is cash + hedge value - option liability marked at Black-Scholes with the
current Heston volatility. The reward is the one-step wealth change with a
mean-variance penalty, r = dW - (kappa / 2) * dW^2, and terminal wealth is the
hedging P&L (premium received minus everything paid out).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .market import HestonParams, bs_call, bs_delta, simulate_heston


@dataclass(frozen=True)
class EnvConfig:
    heston: HestonParams = HestonParams()
    strike: float = 100.0
    n_steps: int = 21          # one trading month, daily rebalancing
    cost_prop: float = 0.001   # 10 bp of traded notional
    cost_impact: float = 0.02  # impact cost = cost_impact * (shares traded)^2 * price
    max_trade: float = 0.25    # shares per asset per step
    max_hold: float = 1.5
    budget: float = 1.0        # gross hedge notional cap, as a multiple of initial prices
    kappa: float = 0.3         # risk aversion in the reward


class HedgingEnv:
    def __init__(self, cfg: EnvConfig, n_envs: int, generator: torch.Generator | None = None):
        self.cfg, self.n, self.g = cfg, n_envs, generator
        self.k = cfg.heston.n_assets
        self.dt = 1.0 / cfg.heston.steps_per_year

    @property
    def obs_dim(self) -> int:
        return 4 * self.k + 1

    def reset(self) -> torch.Tensor:
        c = self.cfg
        self.S, self.V = simulate_heston(c.heston, self.n, c.n_steps, self.g)
        self.t = 0
        self.h = torch.zeros(self.n, self.k)
        premium = self._liability().sum(dim=1)
        self.cash = premium.clone()
        self.wealth = torch.zeros(self.n)
        self.total_cost = torch.zeros(self.n)
        return self.obs()

    def tau(self, t: int | None = None) -> float:
        t = self.t if t is None else t
        return (self.cfg.n_steps - t) * self.dt

    def vol(self) -> torch.Tensor:
        return torch.sqrt(self.V[:, self.t])

    def _liability(self) -> torch.Tensor:
        s = self.S[:, self.t]
        return bs_call(s, self.cfg.strike, torch.tensor(self.tau()), self.vol())

    def delta(self) -> torch.Tensor:
        return bs_delta(self.S[:, self.t], self.cfg.strike, torch.tensor(self.tau()), self.vol())

    def obs(self) -> torch.Tensor:
        s = self.S[:, self.t]
        frac_left = torch.full((self.n, 1), self.tau() / (self.cfg.n_steps * self.dt))
        return torch.cat([torch.log(s / self.cfg.strike) * 10, self.vol() * 5, self.h,
                          self.delta(), frac_left], dim=1)

    def project(self, target: torch.Tensor) -> torch.Tensor:
        """Constraint layer: trade limit, position box, then budget."""
        c = self.cfg
        a = torch.clamp(target, self.h - c.max_trade, self.h + c.max_trade)
        a = a.clamp(0.0, c.max_hold)
        s = self.S[:, self.t]
        notional = (a * s).sum(dim=1, keepdim=True)
        cap = c.budget * c.heston.s0 * self.k
        scale = torch.clamp(cap / notional.clamp(min=1e-9), max=1.0)
        a = a * scale
        # scaling down must not break the trade limit for positions being reduced
        return torch.clamp(a, self.h - c.max_trade, self.h + c.max_trade).clamp(0.0, c.max_hold)

    def step(self, target: torch.Tensor):
        """Returns (next_obs, reward, done, executed_action)."""
        c = self.cfg
        a = self.project(target)
        s = self.S[:, self.t]
        trade = a - self.h
        cost = (c.cost_prop * trade.abs() * s + c.cost_impact * trade**2 * s).sum(dim=1)
        self.cash = self.cash - (trade * s).sum(dim=1) - cost
        self.total_cost += cost
        self.h = a
        self.t += 1
        new_wealth = self.cash + (self.h * self.S[:, self.t]).sum(dim=1) - self._liability().sum(dim=1)
        dw = new_wealth - self.wealth
        self.wealth = new_wealth
        reward = dw - 0.5 * c.kappa * dw**2
        done = self.t >= c.n_steps
        return self.obs(), reward, done, a
