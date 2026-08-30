"""Feed-forward neural network for stock direction classification."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class DirectionClassifier(nn.Module):
    """A small MLP predicting the logit for "price rises within the horizon"."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...] = (32, 16)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(previous, size))
            layers.append(nn.ReLU())
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid); pair with ``BCEWithLogitsLoss`` for training."""

        return cast(torch.Tensor, self.network(x).squeeze(-1))
