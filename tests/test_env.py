import torch

from hedging.env import EnvConfig, HedgingEnv


def _env(n=512, **kw):
    return HedgingEnv(EnvConfig(**kw), n, torch.Generator().manual_seed(0))


def test_wealth_starts_at_zero_and_terminal_is_payoff_based():
    env = _env()
    env.reset()
    assert torch.allclose(env.wealth, torch.zeros(env.n))
    while True:
        _, _, done, _ = env.step(torch.zeros(env.n, env.k))
        if done:
            break
    # no hedge: P&L = premium - payoff
    payoff = (env.S[:, -1] - env.cfg.strike).clamp(min=0).sum(dim=1)
    premium = env.cash  # no trades, cash is just the premium
    assert torch.allclose(env.wealth, premium - payoff, atol=1e-4)


def test_constraints_hold():
    env = _env(max_trade=0.1, budget=0.5)
    env.reset()
    for _ in range(env.cfg.n_steps):
        before = env.h.clone()
        s = env.S[:, env.t]
        _, _, done, a = env.step(torch.full((env.n, env.k), 5.0))
        assert ((a - before).abs() <= 0.1 + 1e-6).all()
        assert (a >= 0).all() and (a <= env.cfg.max_hold + 1e-6).all()
        assert ((a * s).sum(dim=1) <= 0.5 * 100 * env.k + 1e-3).all()
        if done:
            break


def test_costs_are_charged_on_trades():
    env = _env(cost_prop=0.01, cost_impact=0.0)
    env.reset()
    s0 = env.S[:, 0].clone()
    env.step(torch.full((env.n, env.k), 0.2))
    assert torch.allclose(env.total_cost, (0.01 * 0.2 * s0).sum(dim=1), atol=1e-4)


def test_delta_hedging_reduces_pnl_variance():
    def run(policy):
        env = _env(n=4000, cost_prop=0.0, cost_impact=0.0, max_trade=1.5, budget=2.0)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(policy(env))
        return env.wealth.std()

    unhedged = run(lambda e: torch.zeros(e.n, e.k))
    hedged = run(lambda e: e.delta())
    assert hedged < 0.5 * unhedged
