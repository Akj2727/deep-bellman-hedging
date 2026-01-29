"""Train DDPG and TD3, then score them against the baselines on fresh paths.

    python -m hedging.train --episodes 12800

Episodes run 64 at a time on freshly simulated Heston paths. Before learning, the
replay buffer is warmed up with noisy delta-hedge trajectories so early updates see
sensible actions.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .agents import Agent, AgentConfig, Replay
from .env import EnvConfig, HedgingEnv
from .strategies import agent_policy, delta_hedge, evaluate, no_hedge

RESULTS = Path(__file__).resolve().parents[2] / "results"


def train(agent: Agent, cfg: EnvConfig, episodes: int, n_envs: int = 64,
          updates_per_step: int = 2, warmup_batches: int = 20, seed: int = 0,
          log_every: int = 25) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    env = HedgingEnv(cfg, n_envs, g)
    replay = Replay(env.obs_dim, env.k)
    log = []
    n_batches = episodes // n_envs
    for b in range(warmup_batches + n_batches):
        warm = b < warmup_batches
        frac = max(0, b - warmup_batches) / max(1, n_batches)
        noise = agent.cfg.explore_noise * (1 - 0.8 * frac)
        obs = env.reset()
        done = False
        while not done:
            if warm:
                target = env.delta() + torch.randn(n_envs, env.k, generator=g) * 0.2
            else:
                target = agent.act(obs, noise)
            obs2, r, done, a = env.step(target)
            replay.add(obs, a, r, obs2, done)
            obs = obs2
            if not warm:
                for _ in range(updates_per_step):
                    agent.update(replay)
        if not warm and (b - warmup_batches) % log_every == 0:
            ev = evaluate(agent_policy(agent), cfg, n_paths=2000, seed=999)
            ev["episodes"] = (b - warmup_batches + 1) * n_envs
            log.append(ev)
            print(f"{agent.name} ep {ev['episodes']:>6}  mean-var {ev['mean_variance']:.4f}  "
                  f"std {ev['std_pnl']:.3f}  cost {ev['mean_cost']:.3f}", flush=True)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12_800)
    ap.add_argument("--eval-paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    cfg = EnvConfig()
    results, curves = {}, {}
    for name, fn in [("No hedge", no_hedge), ("Delta hedge", delta_hedge)]:
        results[name] = evaluate(fn, cfg, args.eval_paths)
    for td3 in (False, True):
        agent = Agent(HedgingEnv(cfg, 1).obs_dim, cfg.heston.n_assets, cfg.max_hold,
                      AgentConfig(td3=td3), seed=args.seed)
        t0 = time.time()
        curves[agent.name] = train(agent, cfg, args.episodes, seed=args.seed)
        res = evaluate(agent_policy(agent), cfg, args.eval_paths)
        res["train_seconds"] = round(time.time() - t0, 1)
        results[agent.name] = res
        torch.save(agent.actor.state_dict(), RESULTS / f"{agent.name.lower()}_actor.pt")

    base = results["Delta hedge"]["annual_hedging_loss"]
    for r in results.values():
        r["loss_vs_delta_pct"] = 100 * (r["annual_hedging_loss"] / base - 1)
    (RESULTS / "results.json").write_text(json.dumps(
        {"config": {"episodes": args.episodes, "eval_paths": args.eval_paths},
         "results": results, "curves": curves}, indent=2))
    for k, v in results.items():
        print(f"{k:12s} " + "  ".join(f"{m} {x:.4f}" for m, x in v.items()))


if __name__ == "__main__":
    main()
