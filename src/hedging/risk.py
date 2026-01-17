"""Scoring terminal hedging P&L."""
from __future__ import annotations

import torch

MONTHS_PER_YEAR = 12  # each episode hedges a one-month option


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.95) -> float:
    """Mean loss in the worst (1 - alpha) share of paths."""
    k = max(1, int(round((1 - alpha) * pnl.numel())))
    return float(-torch.topk(pnl, k, largest=False).values.mean())


def summarize(pnl: torch.Tensor, kappa: float) -> dict:
    mean, std = float(pnl.mean()), float(pnl.std())
    # Hedging loss per episode = expected shortfall of terminal P&L (the cost of carrying
    # the hedged book in a bad month); annualised over 12 monthly rolls.
    return {
        "mean_pnl": mean,
        "std_pnl": std,
        "cvar95_loss": cvar_loss(pnl),
        "mean_variance": mean - 0.5 * kappa * std**2,
        "annual_hedging_loss": -(mean - 0.5 * kappa * std**2) * MONTHS_PER_YEAR,
    }
