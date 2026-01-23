"""Baseline hedges and a common evaluation loop."""
from __future__ import annotations

from typing import Callable

import torch

from .env import EnvConfig, HedgingEnv
from .risk import summarize

Policy = Callable[[HedgingEnv, torch.Tensor], torch.Tensor]


def no_hedge(env: HedgingEnv, obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(env.n, env.k)


def delta_hedge(env: HedgingEnv, obs: torch.Tensor) -> torch.Tensor:
    """Black-Scholes delta at the current Heston vol, sent through the same constraints."""
    return env.delta()


def agent_policy(agent) -> Policy:
    return lambda env, obs: agent.act(obs)


@torch.no_grad()
def evaluate(policy: Policy, cfg: EnvConfig, n_paths: int = 10_000, seed: int = 12345) -> dict:
    env = HedgingEnv(cfg, n_paths, torch.Generator().manual_seed(seed))
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(policy(env, obs))
    out = summarize(env.wealth, cfg.kappa)
    out["mean_cost"] = float(env.total_cost.mean())
    return out
