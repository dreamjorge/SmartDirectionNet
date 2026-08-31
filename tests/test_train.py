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


def _separable_sequence_dataset(n_sequences, window, seed):
    from smartdirectionnet.features import SequenceDataset

    rng = np.random.default_rng(seed)
    # Label depends only on the mean of the sequence's single feature: separable and
    # trivially learnable by an LSTM regardless of window ordering.
    x = rng.normal(size=(n_sequences, window, 1)).astype("float32")
    y = (x.mean(axis=1)[:, 0] > 0).astype("float32")
    tickers = np.array(["A"] * n_sequences, dtype=object)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n_sequences), unit="D")
    return SequenceDataset(
        X=x,
        y=y,
        tickers=tickers,
        feature_columns=["feature"],
        window=window,
        anchor_dates=dates.to_numpy(),
        label_dates=(dates + pd.Timedelta(days=1)).to_numpy(),
    )


def test_train_sequence_classifier_learns_a_separable_pattern():
    from smartdirectionnet.train import train_sequence_classifier

    train_dataset = _separable_sequence_dataset(200, window=5, seed=0)
    test_dataset = _separable_sequence_dataset(50, window=5, seed=1)

    trained, metrics = train_sequence_classifier(
        train_dataset, test_dataset, hidden_size=8, epochs=60, lr=0.05, seed=0
    )

    assert metrics["train_accuracy"] > 0.85
    assert metrics["test_accuracy"] > 0.85


def test_train_sequence_classifier_rejects_empty_datasets():
    from smartdirectionnet.features import SequenceDataset
    from smartdirectionnet.train import train_sequence_classifier

    empty = SequenceDataset(
        X=np.empty((0, 3, 1), dtype="float32"),
        y=np.empty((0,), dtype="float32"),
        tickers=np.empty((0,), dtype=object),
        feature_columns=["feature"],
        window=3,
        anchor_dates=np.empty((0,)),
        label_dates=np.empty((0,)),
    )
    non_empty = _separable_sequence_dataset(10, window=3, seed=0)

    with pytest.raises(ValueError, match="non-empty"):
        train_sequence_classifier(empty, non_empty)


def test_save_and_load_sequence_model_round_trips_identical_predictions(tmp_path):
    from smartdirectionnet.train import (
        load_sequence_model,
        predict_sequence,
        save_sequence_model,
        train_sequence_classifier,
    )

    train_dataset = _separable_sequence_dataset(100, window=4, seed=0)
    test_dataset = _separable_sequence_dataset(20, window=4, seed=1)
    trained, _ = train_sequence_classifier(
        train_dataset, test_dataset, hidden_size=8, epochs=20, seed=0
    )
    model_path = tmp_path / "model.pt"

    before = predict_sequence(trained, test_dataset)
    save_sequence_model(trained, model_path)
    reloaded = load_sequence_model(model_path)
    after = predict_sequence(reloaded, test_dataset)

    assert reloaded.feature_columns == trained.feature_columns
    assert reloaded.window == trained.window
    assert np.allclose(before, after)


def test_predict_sequence_reorders_mismatched_feature_column_order():
    from smartdirectionnet.features import SequenceDataset
    from smartdirectionnet.train import predict_sequence, train_sequence_classifier

    rng = np.random.default_rng(0)
    n = 50
    feature_a = rng.normal(size=(n, 4, 1)).astype("float32")
    feature_b = rng.normal(size=(n, 4, 1)).astype("float32")
    x_train = np.concatenate([feature_a, feature_b], axis=2)
    y_train = (feature_a.mean(axis=1)[:, 0] > 0).astype("float32")
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    train_dataset = SequenceDataset(
        X=x_train,
        y=y_train,
        tickers=np.array(["A"] * n, dtype=object),
        feature_columns=["a", "b"],
        window=4,
        anchor_dates=dates.to_numpy(),
        label_dates=(dates + pd.Timedelta(days=1)).to_numpy(),
    )
    trained, _ = train_sequence_classifier(
        train_dataset, train_dataset, hidden_size=8, epochs=5, seed=0
    )

    swapped_dataset = SequenceDataset(
        X=x_train[:, :, [1, 0]],
        y=y_train,
        tickers=train_dataset.tickers,
        feature_columns=["b", "a"],
        window=4,
        anchor_dates=train_dataset.anchor_dates,
        label_dates=train_dataset.label_dates,
    )

    same_order_predictions = predict_sequence(trained, train_dataset)
    reordered_predictions = predict_sequence(trained, swapped_dataset)

    assert np.allclose(same_order_predictions, reordered_predictions)


def test_predict_sequence_rejects_mismatched_feature_set():
    from smartdirectionnet.features import SequenceDataset
    from smartdirectionnet.train import predict_sequence, train_sequence_classifier

    train_dataset = _separable_sequence_dataset(20, window=3, seed=0)
    trained, _ = train_sequence_classifier(
        train_dataset, train_dataset, hidden_size=4, epochs=5, seed=0
    )
    wrong_dataset = SequenceDataset(
        X=train_dataset.X,
        y=train_dataset.y,
        tickers=train_dataset.tickers,
        feature_columns=["different_feature"],
        window=train_dataset.window,
        anchor_dates=train_dataset.anchor_dates,
        label_dates=train_dataset.label_dates,
    )

    with pytest.raises(ValueError, match="feature_columns"):
        predict_sequence(trained, wrong_dataset)
