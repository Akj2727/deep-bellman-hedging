import math

import torch

from hedging.market import HestonParams, bs_call, bs_delta, simulate_heston


def test_bs_call_known_value():
    # S = K = 100, 1 year, 20% vol, zero rates -> about 7.9656
    p = bs_call(torch.tensor(100.0), 100.0, torch.tensor(1.0), torch.tensor(0.2))
    assert abs(p.item() - 7.9656) < 1e-3


def test_bs_delta_matches_finite_difference():
    f64 = dict(dtype=torch.float64)
    s, tau, vol = torch.tensor(103.0, **f64), torch.tensor(0.1, **f64), torch.tensor(0.25, **f64)
    h = 1e-4
    fd = (bs_call(s + h, 100.0, tau, vol) - bs_call(s - h, 100.0, tau, vol)) / (2 * h)
    assert abs(fd.item() - bs_delta(s, 100.0, tau, vol).item()) < 1e-6


def test_heston_is_a_martingale_and_variance_mean_reverts():
    g = torch.Generator().manual_seed(0)
    p = HestonParams()
    S, V = simulate_heston(p, 20000, 252, g)
    mean_end = S[:, -1].mean(dim=0)
    assert torch.all((mean_end - p.s0).abs() < 1.0)  # zero drift, within MC error
    assert abs(V[:, -1].mean().item() - p.theta) < 0.005
    assert (V >= 0).all()


def test_asset_correlation():
    g = torch.Generator().manual_seed(1)
    S, _ = simulate_heston(HestonParams(), 20000, 1, g)
    r = torch.log(S[:, 1] / S[:, 0])
    c = torch.corrcoef(r.T)[0, 1].item()
    assert abs(c - 0.5) < 0.03
