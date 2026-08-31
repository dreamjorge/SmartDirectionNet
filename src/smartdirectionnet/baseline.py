"""LightGBM gradient-boosted tree baseline (not a neural network).

Gradient-boosted trees are the standard baseline for tabular financial data and often
outperform neural networks on this kind of data. Train one on the same point-in-time
dataset the MLP uses, to have an honest comparison point for the MLP/LSTM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass
class TrainedBaseline:
    """A trained gradient-boosted tree baseline paired with the feature list it expects."""

    model: lgb.Booster
    feature_columns: list[str]


def train_gbm_baseline(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    num_boost_round: int = 100,
    seed: int = 0,
    **lgb_params: Any,
) -> tuple[TrainedBaseline, dict[str, float]]:
    """Train a LightGBM binary classifier and return it with train/test accuracy."""

    if feature_columns is None:
        feature_columns = train_frame.attrs.get("feature_columns")
    if not feature_columns:
        raise ValueError("feature_columns must be provided or present in train_frame.attrs")
    if train_frame.empty or test_frame.empty:
        raise ValueError("train_frame and test_frame must both be non-empty")

    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "seed": seed,
        **lgb_params,
    }

    train_data = lgb.Dataset(train_frame[feature_columns], label=train_frame["label"])
    booster = lgb.train(params, train_data, num_boost_round=num_boost_round)

    train_pred = (np.asarray(booster.predict(train_frame[feature_columns])) > 0.5).astype(int)
    test_pred = (np.asarray(booster.predict(test_frame[feature_columns])) > 0.5).astype(int)

    train_accuracy = float((train_pred == train_frame["label"].to_numpy()).mean())
    test_accuracy = float((test_pred == test_frame["label"].to_numpy()).mean())

    trained = TrainedBaseline(model=booster, feature_columns=list(feature_columns))
    metrics = {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}
    return trained, metrics


def predict_gbm_baseline(trained: TrainedBaseline, frame: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability of a price rise for each row in ``frame``."""

    return np.asarray(trained.model.predict(frame[trained.feature_columns]))


def save_gbm_baseline(trained: TrainedBaseline, path: str | Path) -> None:
    """Save a trained baseline's booster and feature list as a single JSON file."""

    payload = {
        "feature_columns": trained.feature_columns,
        "model_string": trained.model.model_to_string(),
    }
    Path(path).write_text(json.dumps(payload))


def load_gbm_baseline(path: str | Path) -> TrainedBaseline:
    """Load a baseline saved by ``save_gbm_baseline``."""

    payload = json.loads(Path(path).read_text())
    booster = lgb.Booster(model_str=payload["model_string"])
    return TrainedBaseline(model=booster, feature_columns=payload["feature_columns"])
