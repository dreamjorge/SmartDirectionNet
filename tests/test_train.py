import numpy as np
import pandas as pd
import pytest

from smartdirectionnet.train import load_model, predict, save_model, train_direction_classifier


def _separable_frame(n, seed):
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=n)
    label = (feature > 0).astype(int)
    return pd.DataFrame({"feature": feature, "label": label})


def test_train_direction_classifier_learns_a_linearly_separable_pattern():
    train_frame = _separable_frame(200, seed=0)
    test_frame = _separable_frame(50, seed=1)

    trained, metrics = train_direction_classifier(
        train_frame,
        test_frame,
        feature_columns=["feature"],
        hidden_sizes=(8,),
        epochs=200,
        lr=0.05,
        seed=0,
    )

    assert metrics["train_accuracy"] > 0.9
    assert metrics["test_accuracy"] > 0.9


def test_train_direction_classifier_rejects_empty_frames():
    empty = pd.DataFrame({"feature": [], "label": []})
    non_empty = _separable_frame(10, seed=0)

    with pytest.raises(ValueError, match="non-empty"):
        train_direction_classifier(empty, non_empty, feature_columns=["feature"])


def test_train_direction_classifier_requires_feature_columns():
    frame = _separable_frame(10, seed=0)

    with pytest.raises(ValueError, match="feature_columns"):
        train_direction_classifier(frame, frame)


def test_save_and_load_model_round_trips_identical_predictions(tmp_path):
    train_frame = _separable_frame(100, seed=0)
    test_frame = _separable_frame(20, seed=1)
    trained, _ = train_direction_classifier(
        train_frame, test_frame, feature_columns=["feature"], epochs=20, seed=0
    )
    model_path = tmp_path / "model.pt"

    before = predict(trained, test_frame)
    save_model(trained, model_path)
    reloaded = load_model(model_path)
    after = predict(reloaded, test_frame)

    assert reloaded.feature_columns == trained.feature_columns
    assert np.allclose(before, after)
