"""Feature engineering: turn enriched OHLCV rows into a supervised direction dataset."""

from __future__ import annotations

import pandas as pd

_NON_FEATURE_COLUMNS = {"date", "open", "high", "low", "close", "volume", "ticker", "label"}


def build_direction_dataset(
    frame: pd.DataFrame,
    *,
    horizon: int = 5,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return a labeled dataset for next-``horizon``-row price direction classification.

    ``frame`` must already be cleaned, indicator-enriched, and sorted by (ticker, date) —
    exactly the shape returned by ``smartanalyticsinvest.pipeline.enrich_ohlcv``. Adds a
    ``label`` column (``1`` if ``close`` is higher ``horizon`` rows ahead, else ``0``).

    Rows with no future row to label (the last ``horizon`` rows of each ticker) and rows
    with any missing feature value (e.g. an indicator's rolling-window warm-up period) are
    dropped. Labels never use another ticker's future rows.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    group_keys = frame["ticker"] if "ticker" in frame.columns else pd.Series(0, index=frame.index)
    future_close = frame.groupby(group_keys, sort=False)["close"].shift(-horizon)

    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column not in _NON_FEATURE_COLUMNS]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No feature columns available to build the dataset")

    valid_mask = future_close.notna() & frame[feature_columns].notna().all(axis=1)

    result = frame.loc[valid_mask].copy()
    result["label"] = (future_close.loc[valid_mask] > frame.loc[valid_mask, "close"]).astype(int)
    result = result.reset_index(drop=True)
    result.attrs["feature_columns"] = feature_columns
    return result


def time_series_split(
    frame: pd.DataFrame, *, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a chronological (non-shuffled) train/test split, per ticker when present.

    Unlike a random split, this guarantees every training row predates every test row
    for the same ticker, avoiding look-ahead leakage from the future into training data.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    if "ticker" not in frame.columns:
        cutoff = int(len(frame) * (1 - test_size))
        train = frame.iloc[:cutoff].reset_index(drop=True)
        test = frame.iloc[cutoff:].reset_index(drop=True)
    else:
        train_parts = []
        test_parts = []
        for _, group in frame.groupby("ticker", sort=False):
            cutoff = int(len(group) * (1 - test_size))
            train_parts.append(group.iloc[:cutoff])
            test_parts.append(group.iloc[cutoff:])
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)

    train.attrs["feature_columns"] = frame.attrs.get("feature_columns")
    test.attrs["feature_columns"] = frame.attrs.get("feature_columns")
    return train, test
