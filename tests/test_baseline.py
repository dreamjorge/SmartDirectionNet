import numpy as np
import pandas as pd
import pytest

from smartdirectionnet.baseline import (
    load_gbm_baseline,
    predict_gbm_baseline,
    save_gbm_baseline,
    train_gbm_baseline,
)


def _separable_frame(n, seed):
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=n)
    label = (feature > 0).astype(int)
    return pd.DataFrame({"feature": feature, "label": label})


def test_train_gbm_baseline_learns_a_linearly_separable_pattern():
    train_frame = _separable_frame(200, seed=0)
    test_frame = _separable_frame(50, seed=1)

    trained, metrics = train_gbm_baseline(
        train_frame, test_frame, feature_columns=["feature"], num_boost_round=20
    )

    assert metrics["train_accuracy"] > 0.9
    assert metrics["test_accuracy"] > 0.9


def test_train_gbm_baseline_rejects_empty_frames():
    empty = pd.DataFrame({"feature": [], "label": []})
    non_empty = _separable_frame(10, seed=0)

    with pytest.raises(ValueError, match="non-empty"):
        train_gbm_baseline(empty, non_empty, feature_columns=["feature"])


def test_train_gbm_baseline_requires_feature_columns():
    frame = _separable_frame(10, seed=0)

    with pytest.raises(ValueError, match="feature_columns"):
        train_gbm_baseline(frame, frame)


def test_save_and_load_gbm_baseline_round_trips_identical_predictions(tmp_path):
    train_frame = _separable_frame(100, seed=0)
    test_frame = _separable_frame(20, seed=1)
    trained, _ = train_gbm_baseline(
        train_frame, test_frame, feature_columns=["feature"], num_boost_round=20
    )
    model_path = tmp_path / "model.json"

    before = predict_gbm_baseline(trained, test_frame)
    save_gbm_baseline(trained, model_path)
    reloaded = load_gbm_baseline(model_path)
    after = predict_gbm_baseline(reloaded, test_frame)

    assert reloaded.feature_columns == trained.feature_columns
    assert np.allclose(before, after)
