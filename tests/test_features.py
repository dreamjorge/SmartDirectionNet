import pandas as pd
import pytest

from smartdirectionnet.features import build_direction_dataset, time_series_split


def _frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "open": [1] * 6,
            "high": [1] * 6,
            "low": [1] * 6,
            "close": [10, 11, 9, 12, 8, 15],
            "volume": [100] * 6,
            "sma_2": [None, 10.5, 10, 10.5, 10, 11.5],
        }
    )


def test_build_direction_dataset_labels_future_close_and_drops_incomplete_rows():
    result = build_direction_dataset(_frame(), horizon=2, feature_columns=["sma_2"])

    assert result["label"].tolist() == [1, 0, 1]
    assert result["close"].tolist() == [11, 9, 12]
    assert result.attrs["feature_columns"] == ["sma_2"]


def test_build_direction_dataset_defaults_to_indicator_columns():
    frame = _frame().rename(columns={"sma_2": "rsi_14"})

    result = build_direction_dataset(frame, horizon=2)

    assert result.attrs["feature_columns"] == ["rsi_14"]


def test_build_direction_dataset_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="horizon"):
        build_direction_dataset(_frame(), horizon=0, feature_columns=["sma_2"])


def test_build_direction_dataset_rejects_no_feature_columns():
    frame = _frame().drop(columns=["sma_2"])

    with pytest.raises(ValueError, match="feature columns"):
        build_direction_dataset(frame, horizon=1)


def test_build_direction_dataset_labels_never_cross_ticker_boundaries():
    frame = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=3, freq="D")) * 2,
            "close": [10, 11, 12, 100, 90, 80],
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "sma_2": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    result = build_direction_dataset(frame, horizon=1, feature_columns=["sma_2"])

    # A's last row and B's last row have no future row within their own ticker and
    # are dropped; none of A's labels should be computed against B's prices.
    assert result[["ticker", "close", "label"]].to_dict("records") == [
        {"ticker": "A", "close": 10, "label": 1},
        {"ticker": "A", "close": 11, "label": 1},
        {"ticker": "B", "close": 100, "label": 0},
        {"ticker": "B", "close": 90, "label": 0},
    ]


def test_time_series_split_is_chronological_without_ticker():
    frame = pd.DataFrame({"date": range(10), "close": range(10)})

    train, test = time_series_split(frame, test_size=0.3)

    assert train["date"].tolist() == list(range(7))
    assert test["date"].tolist() == list(range(7, 10))


def test_time_series_split_splits_each_ticker_independently_without_leakage():
    frame = pd.DataFrame(
        {
            "date": list(range(5)) * 2,
            "ticker": ["A"] * 5 + ["B"] * 5,
        }
    )

    train, test = time_series_split(frame, test_size=0.4)

    for ticker in ("A", "B"):
        train_dates = train.loc[train["ticker"] == ticker, "date"]
        test_dates = test.loc[test["ticker"] == ticker, "date"]
        assert train_dates.max() < test_dates.min()


def test_time_series_split_rejects_invalid_test_size():
    frame = pd.DataFrame({"date": range(5)})

    with pytest.raises(ValueError, match="test_size"):
        time_series_split(frame, test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        time_series_split(frame, test_size=1.0)


def _sequence_frame(feature=None):
    close = [10, 11, 12, 13, 9, 20]
    return pd.DataFrame(
        {
            "close": close,
            "sma_2": feature if feature is not None else [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_build_sequence_dataset_windows_and_labels_match_hand_computation():
    from smartdirectionnet.features import build_sequence_dataset

    dataset = build_sequence_dataset(
        _sequence_frame(), window=3, horizon=1, feature_columns=["sma_2"]
    )

    assert dataset.X.shape == (3, 3, 1)
    assert dataset.X[:, :, 0].tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert dataset.y.tolist() == [1.0, 0.0, 1.0]
    assert dataset.feature_columns == ["sma_2"]
    assert dataset.window == 3


def test_build_sequence_dataset_drops_windows_with_missing_feature_values():
    from smartdirectionnet.features import build_sequence_dataset

    frame = _sequence_frame(feature=[1.0, None, 3.0, 4.0, 5.0, 6.0])

    dataset = build_sequence_dataset(frame, window=3, horizon=1, feature_columns=["sma_2"])

    # windows anchored at i=2 ([0,1,2]) and i=3 ([1,2,3]) both touch the missing index 1
    assert dataset.X.shape == (1, 3, 1)
    assert dataset.X[0, :, 0].tolist() == [3.0, 4.0, 5.0]
    assert dataset.y.tolist() == [1.0]


def test_build_sequence_dataset_windows_never_cross_ticker_boundaries():
    from smartdirectionnet.features import build_sequence_dataset

    frame = pd.DataFrame(
        {
            "close": [10, 11, 12, 13, 100, 101, 102, 103],
            "sma_2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )

    dataset = build_sequence_dataset(frame, window=3, horizon=1, feature_columns=["sma_2"])

    # 4 rows per ticker with window=3, horizon=1 leaves exactly one valid anchor per ticker
    assert dataset.X.shape == (2, 3, 1)
    assert sorted(dataset.tickers.tolist()) == ["A", "B"]
    for ticker, expected_features in [("A", [1.0, 2.0, 3.0]), ("B", [5.0, 6.0, 7.0])]:
        index = dataset.tickers.tolist().index(ticker)
        assert dataset.X[index, :, 0].tolist() == expected_features
        assert dataset.y[index] == 1.0


def test_build_sequence_dataset_rejects_non_positive_window_or_horizon():
    from smartdirectionnet.features import build_sequence_dataset

    with pytest.raises(ValueError, match="window"):
        build_sequence_dataset(_sequence_frame(), window=0, feature_columns=["sma_2"])
    with pytest.raises(ValueError, match="horizon"):
        build_sequence_dataset(_sequence_frame(), horizon=0, feature_columns=["sma_2"])


def test_build_sequence_dataset_raises_when_no_samples_fit():
    from smartdirectionnet.features import build_sequence_dataset

    with pytest.raises(ValueError, match="No samples"):
        build_sequence_dataset(_sequence_frame(), window=10, horizon=1, feature_columns=["sma_2"])


def test_sequence_time_series_split_has_no_leakage_per_ticker():
    from smartdirectionnet.features import build_sequence_dataset, sequence_time_series_split

    rows = 12
    frame = pd.DataFrame(
        {
            "close": list(range(rows)) + list(range(100, 100 + rows)),
            "sma_2": list(range(rows)) + list(range(100, 100 + rows)),
            "ticker": ["A"] * rows + ["B"] * rows,
        }
    )
    dataset = build_sequence_dataset(frame, window=3, horizon=1, feature_columns=["sma_2"])

    train, test = sequence_time_series_split(dataset, test_size=0.25)

    assert len(train.y) + len(test.y) == len(dataset.y)
    for ticker in ("A", "B"):
        train_last_values = train.X[train.tickers == ticker][:, -1, 0]
        test_last_values = test.X[test.tickers == ticker][:, -1, 0]
        assert train_last_values.max() < test_last_values.min()


def test_sequence_time_series_split_rejects_invalid_test_size():
    from smartdirectionnet.features import build_sequence_dataset, sequence_time_series_split

    dataset = build_sequence_dataset(
        _sequence_frame(), window=3, horizon=1, feature_columns=["sma_2"]
    )

    with pytest.raises(ValueError, match="test_size"):
        sequence_time_series_split(dataset, test_size=0.0)
