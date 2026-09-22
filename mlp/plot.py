from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_curve(
    run_name: str,
    history: Sequence[Dict[str, float]],
    select: Dict[str, Any],
    best_epoch: int,
    path: Path,
) -> None:
    metric = select["metric"]
    epochs = [record["epoch"] for record in history]
    losses = [record["train_loss"] for record in history]
    values = [record[metric] for record in history]
    best_value = values[best_epoch - 1]

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    top.plot(epochs, losses, color="tab:blue")
    top.set_ylabel("train loss")
    top.set_title(f"{run_name} — best epoch {best_epoch}, {metric}={best_value:.4f}")

    bottom.plot(epochs, values, color="tab:orange")
    bottom.set_ylabel(f"val {metric}")
    bottom.set_xlabel("epoch")
    bottom.plot([best_epoch], [best_value], marker="o", color="tab:red")

    for axis in (top, bottom):
        axis.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
