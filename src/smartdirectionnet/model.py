"""Neural network architectures for stock direction classification."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class DirectionClassifier(nn.Module):
    """A small MLP predicting the logit for "price rises within the horizon" from a
    single row's feature snapshot."""

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


class DirectionSequenceClassifier(nn.Module):
    """An LSTM predicting the logit for "price rises within the horizon" from a
    trailing window of feature rows, shaped ``(batch, window, input_dim)``."""

    def __init__(self, input_dim: int, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid) from the LSTM's final hidden state."""

        _, (hidden, _) = self.lstm(x)
        last_layer_hidden = hidden[-1]
        return cast(torch.Tensor, self.head(last_layer_hidden).squeeze(-1))
