"""Training loop, persistence, and inference for DirectionClassifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from smartdirectionnet.model import DirectionClassifier


@dataclass
class TrainedModel:
    """A trained model paired with the feature list and normalization stats it expects."""

    model: DirectionClassifier
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _to_tensor(
    frame: pd.DataFrame, feature_columns: list[str], mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    values = frame[feature_columns].to_numpy(dtype="float32")
    normalized = (values - mean) / std
    return torch.from_numpy(normalized.astype("float32"))


def train_direction_classifier(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    hidden_sizes: tuple[int, ...] = (32, 16),
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[TrainedModel, dict[str, float]]:
    """Train a ``DirectionClassifier`` and return it with train/test accuracy."""

    if feature_columns is None:
        feature_columns = train_frame.attrs.get("feature_columns")
    if not feature_columns:
        raise ValueError("feature_columns must be provided or present in train_frame.attrs")
    if train_frame.empty or test_frame.empty:
        raise ValueError("train_frame and test_frame must both be non-empty")

    torch.manual_seed(seed)

    raw_values = train_frame[feature_columns].to_numpy(dtype="float32")
    mean = raw_values.mean(axis=0)
    std = raw_values.std(axis=0)
    std[std == 0] = 1.0

    x_train = _to_tensor(train_frame, feature_columns, mean, std)
    y_train = torch.from_numpy(train_frame["label"].to_numpy(dtype="float32"))
    x_test = _to_tensor(test_frame, feature_columns, mean, std)
    y_test = torch.from_numpy(test_frame["label"].to_numpy(dtype="float32"))

    model = DirectionClassifier(input_dim=len(feature_columns), hidden_sizes=hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_accuracy = (
            ((torch.sigmoid(model(x_train)) > 0.5).float() == y_train).float().mean().item()
        )
        test_accuracy = (
            ((torch.sigmoid(model(x_test)) > 0.5).float() == y_test).float().mean().item()
        )

    trained = TrainedModel(
        model=model, feature_columns=list(feature_columns), feature_mean=mean, feature_std=std
    )
    metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}
    return trained, metrics


def predict(trained: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of a price rise for each row in ``frame``."""

    trained.model.eval()
    x = _to_tensor(frame, trained.feature_columns, trained.feature_mean, trained.feature_std)
    with torch.no_grad():
        probabilities = torch.sigmoid(trained.model(x))
    return probabilities.numpy()


def _hidden_sizes_from_model(model: DirectionClassifier) -> list[int]:
    return [
        layer.out_features
        for layer in model.network
        if isinstance(layer, nn.Linear) and layer.out_features != 1
    ]


def save_model(trained: TrainedModel, path: str | Path) -> None:
    """Save a trained model's weights, feature list, and normalization stats."""

    torch.save(
        {
            "state_dict": trained.model.state_dict(),
            "feature_columns": trained.feature_columns,
            "feature_mean": trained.feature_mean.tolist(),
            "feature_std": trained.feature_std.tolist(),
            "hidden_sizes": _hidden_sizes_from_model(trained.model),
        },
        Path(path),
    )


def load_model(path: str | Path) -> TrainedModel:
    """Load a model saved by ``save_model``."""

    checkpoint = torch.load(Path(path), weights_only=False)
    model = DirectionClassifier(
        input_dim=len(checkpoint["feature_columns"]),
        hidden_sizes=tuple(checkpoint["hidden_sizes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return TrainedModel(
        model=model,
        feature_columns=checkpoint["feature_columns"],
        feature_mean=np.array(checkpoint["feature_mean"], dtype="float32"),
        feature_std=np.array(checkpoint["feature_std"], dtype="float32"),
    )
