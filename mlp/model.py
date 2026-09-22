from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
        dropout: float,
        batch_norm: bool,
        heads: Dict[str, int],
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation!r} (registered: {sorted(ACTIVATIONS)})"
            )

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_dim, dim) for name, dim in heads.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}
