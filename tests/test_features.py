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
