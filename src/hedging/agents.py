"""DDPG and TD3 for continuous hedge ratios.

Both share an actor that maps observations to target holdings in [0, max_hold] and
critics Q(s, a). Transitions store the executed (constraint-projected) action, so the
critics learn the value of what was actually traded.

TD3 adds twin critics (min of the two in the target), delayed actor updates and
target-policy smoothing on top of DDPG.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


def mlp(inp: int, out: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_hold: float):
        super().__init__()
        self.net = mlp(obs_dim, act_dim)
        self.max_hold = max_hold

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs)) * self.max_hold


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=1)).squeeze(1)


class Replay:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 500_000):
        self.o = torch.zeros(capacity, obs_dim)
        self.a = torch.zeros(capacity, act_dim)
        self.r = torch.zeros(capacity)
        self.o2 = torch.zeros(capacity, obs_dim)
        self.d = torch.zeros(capacity)
        self.cap, self.ptr, self.size = capacity, 0, 0

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (torch.arange(n) + self.ptr) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx] = o, a, r, o2
        self.d[idx] = float(d)
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch: int, g: torch.Generator | None = None):
        idx = torch.randint(0, self.size, (batch,), generator=g)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


@dataclass(frozen=True)
class AgentConfig:
    gamma: float = 1.0           # finite horizon; time-to-maturity is in the state
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    tau: float = 0.005           # Polyak averaging for target networks
    batch: int = 256
    explore_noise: float = 0.15  # Gaussian, as a fraction of max_hold
    td3: bool = False
    policy_delay: int = 2
    target_noise: float = 0.1
    target_noise_clip: float = 0.25


class Agent:
    def __init__(self, obs_dim: int, act_dim: int, max_hold: float, cfg: AgentConfig, seed: int = 0):
        torch.manual_seed(seed)
        self.cfg, self.max_hold = cfg, max_hold
        self.actor = Actor(obs_dim, act_dim, max_hold)
        self.actor_t = copy.deepcopy(self.actor)
        n_critics = 2 if cfg.td3 else 1
        self.critics = nn.ModuleList([Critic(obs_dim, act_dim) for _ in range(n_critics)])
        self.critics_t = copy.deepcopy(self.critics)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(self.critics.parameters(), lr=cfg.lr_critic)
        self.updates = 0

    @property
    def name(self) -> str:
        return "TD3" if self.cfg.td3 else "DDPG"

    @torch.no_grad()
    def act(self, obs: torch.Tensor, noise_scale: float = 0.0) -> torch.Tensor:
        a = self.actor(obs)
        if noise_scale > 0:
            a = a + torch.randn_like(a) * noise_scale * self.max_hold
        return a.clamp(0.0, self.max_hold)

    def update(self, replay: Replay) -> dict:
        c = self.cfg
        o, a, r, o2, d = replay.sample(c.batch)
        with torch.no_grad():
            a2 = self.actor_t(o2)
            if c.td3:
                eps = (torch.randn_like(a2) * c.target_noise * self.max_hold).clamp(
                    -c.target_noise_clip * self.max_hold, c.target_noise_clip * self.max_hold)
                a2 = (a2 + eps).clamp(0.0, self.max_hold)
            q2 = torch.stack([q(o2, a2) for q in self.critics_t]).min(dim=0).values
            y = r + c.gamma * (1 - d) * q2
        critic_loss = sum(((q(o, a) - y) ** 2).mean() for q in self.critics)
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        self.updates += 1
        info = {"critic_loss": float(critic_loss)}
        if not c.td3 or self.updates % c.policy_delay == 0:
            actor_loss = -self.critics[0](o, self.actor(o)).mean()
            self.opt_a.zero_grad()
            actor_loss.backward()
            self.opt_a.step()
            self._soft_update(self.actor, self.actor_t)
            self._soft_update(self.critics, self.critics_t)
            info["actor_loss"] = float(actor_loss)
        return info

    @torch.no_grad()
    def _soft_update(self, net: nn.Module, target: nn.Module) -> None:
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)
