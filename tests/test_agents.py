import torch

from hedging.agents import Actor, Agent, AgentConfig, Replay
from hedging.env import EnvConfig, HedgingEnv


def test_untrained_actor_starts_at_delta():
    env = HedgingEnv(EnvConfig(), 256, torch.Generator().manual_seed(0))
    obs = env.reset()
    actor = Actor(env.obs_dim, env.k, env.cfg.max_hold)
    assert torch.allclose(actor(obs), env.delta(), atol=1e-2)


def test_actor_output_bounds():
    actor = Actor(9, 2, 1.5)
    out = actor(torch.randn(1000, 9) * 50)
    assert (out >= 0).all() and (out <= 1.5).all()


def test_replay_wraps_around():
    rb = Replay(3, 2, capacity=10)
    for i in range(4):
        rb.add(torch.full((4, 3), float(i)), torch.zeros(4, 2), torch.zeros(4), torch.zeros(4, 3), False)
    assert rb.size == 10 and rb.ptr == 6
    o, *_ = rb.sample(64)
    assert set(o[:, 0].tolist()) <= {1.0, 2.0, 3.0}


def test_td3_update_runs_and_moves_actor_after_warmup():
    cfg = AgentConfig(td3=True, critic_warmup=5, batch=32)
    agent = Agent(9, 2, 1.5, cfg)
    rb = Replay(9, 2, capacity=1000)
    rb.add(torch.randn(500, 9), torch.rand(500, 2), torch.randn(500), torch.randn(500, 9), False)
    before = [p.clone() for p in agent.actor.parameters()]
    for _ in range(5):
        agent.update(rb)
    assert all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    for _ in range(10):
        agent.update(rb)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
