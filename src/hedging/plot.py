"""Learning curves and test-set comparison from results/results.json.

    python -m hedging.plot
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import RESULTS


def main() -> None:
    d = json.loads((RESULTS / "results.json").read_text())
    res, curves = d["results"], d["curves"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, c in curves.items():
        a1.plot([e["episodes"] for e in c], [e["mean_variance"] for e in c], label=name, lw=1.2)
    a1.axhline(res["Delta hedge"]["mean_variance"], color="gray", ls="--", lw=1, label="Delta hedge")
    a1.set_xlabel("training episodes")
    a1.set_ylabel("mean - (kappa/2) var of P&L (validation)")
    a1.set_title("Learning curves")
    a1.legend(frameon=False)
    a1.grid(alpha=0.3)

    names = [n for n in res if n != "No hedge"]
    a2.bar(names, [res[n]["annual_hedging_loss"] for n in names], color=["#888", "#3b7dd8", "#d8803b"])
    a2.set_ylabel("annualised hedging loss (test, 10,000 paths)")
    a2.set_title("Test set")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "results.png", dpi=130)


if __name__ == "__main__":
    main()
