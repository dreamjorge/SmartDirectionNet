"""Feature engineering: turn enriched OHLCV rows into a supervised direction dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_NON_FEATURE_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ticker",
    "label",
    "_label_date",
}


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

    The result carries a ``_label_date`` column (the future row's date used to compute
    each label), excluded from ``feature_columns``, which ``time_series_split`` uses to
    purge training rows whose label reaches into the test period.
    """

    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    group_keys = frame["ticker"] if "ticker" in frame.columns else pd.Series(0, index=frame.index)
    future_close = frame.groupby(group_keys, sort=False)["close"].shift(-horizon)
    future_date = frame.groupby(group_keys, sort=False)["date"].shift(-horizon)

    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column not in _NON_FEATURE_COLUMNS]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No feature columns available to build the dataset")

    valid_mask = future_close.notna() & frame[feature_columns].notna().all(axis=1)

    result = frame.loc[valid_mask].copy()
    result["label"] = (future_close.loc[valid_mask] > frame.loc[valid_mask, "close"]).astype(int)
    result["_label_date"] = future_date.loc[valid_mask].values
    result = result.reset_index(drop=True)
    result.attrs["feature_columns"] = feature_columns
    return result


def _purge_train_candidates(
    train_candidates: pd.DataFrame, test_start_date: object
) -> pd.DataFrame:
    if "_label_date" not in train_candidates.columns:
        return train_candidates
    safe_mask = train_candidates["_label_date"] < test_start_date
    return train_candidates[safe_mask]


def time_series_split(
    frame: pd.DataFrame, *, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a chronological (non-shuffled) train/test split, per ticker when present.

    Unlike a random split, this guarantees every training row predates every test row
    for the same ticker. If ``frame`` has a ``_label_date`` column (as produced by
    ``build_direction_dataset``), training rows whose label was computed from a date at
    or after the first test row's date are also purged — otherwise, a training label
    could be derived from the same future prices the test set is meant to evaluate on.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    if "ticker" not in frame.columns:
        cutoff = int(len(frame) * (1 - test_size))
        train_candidates = frame.iloc[:cutoff]
        test = frame.iloc[cutoff:].reset_index(drop=True)
        if cutoff < len(frame):
            train_candidates = _purge_train_candidates(train_candidates, frame.iloc[cutoff]["date"])
        train = train_candidates.reset_index(drop=True)
    else:
        train_parts = []
        test_parts = []
        for _, group in frame.groupby("ticker", sort=False):
            cutoff = int(len(group) * (1 - test_size))
            train_candidates = group.iloc[:cutoff]
            if cutoff < len(group):
                train_candidates = _purge_train_candidates(
                    train_candidates, group.iloc[cutoff]["date"]
                )
            train_parts.append(train_candidates)
            test_parts.append(group.iloc[cutoff:])
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)

    train.attrs["feature_columns"] = frame.attrs.get("feature_columns")
    test.attrs["feature_columns"] = frame.attrs.get("feature_columns")
    return train, test


@dataclass
class SequenceDataset:
    """A windowed dataset ready for sequence-model training.

    ``X`` has shape ``(n_samples, window, n_features)``; ``y`` has shape ``(n_samples,)``.
    ``anchor_dates`` is each sample's own (most recent) row date; ``label_dates`` is the
    future date used to compute its label. Both are used by ``sequence_time_series_split``
    to purge training samples whose label reaches into the test period.
    """

    X: np.ndarray
    y: np.ndarray
    tickers: np.ndarray
    feature_columns: list[str]
    window: int
    anchor_dates: np.ndarray
    label_dates: np.ndarray


def build_sequence_dataset(
    frame: pd.DataFrame,
    *,
    window: int = 20,
    horizon: int = 5,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> SequenceDataset:
    """Return a windowed labeled dataset for sequence-model direction classification.

    Like ``build_direction_dataset``, but each sample is the trailing ``window`` rows of
    features (not just the current row), suited to sequence models such as an LSTM.
    ``frame`` must already be cleaned, indicator-enriched, and sorted by (ticker, date).
    A window never spans two different tickers.
    """

    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    if feature_columns is None:
        feature_columns = [column for column in frame.columns if column not in _NON_FEATURE_COLUMNS]
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("No feature columns available to build the dataset")

    has_ticker = "ticker" in frame.columns
    groups = frame.groupby("ticker", sort=False) if has_ticker else [(None, frame)]

    samples: list[np.ndarray] = []
    labels: list[float] = []
    sample_tickers: list[object] = []
    anchor_dates: list[object] = []
    label_dates: list[object] = []

    for ticker_value, group in groups:
        values = group[feature_columns].to_numpy(dtype="float64")
        closes = group["close"].to_numpy(dtype="float64")
        dates = group["date"].to_numpy()
        finite_row = np.isfinite(values).all(axis=1)
        rows = len(group)
        for i in range(window - 1, rows - horizon):
            if not finite_row[i - window + 1 : i + 1].all():
                continue
            samples.append(values[i - window + 1 : i + 1])
            labels.append(float(closes[i + horizon] > closes[i]))
            sample_tickers.append(ticker_value)
            anchor_dates.append(dates[i])
            label_dates.append(dates[i + horizon])

    if not samples:
        raise ValueError(
            "No samples could be built; check that window/horizon fit within each "
            "ticker's row count and that enough feature values are non-missing"
        )

    return SequenceDataset(
        X=np.stack(samples).astype("float32"),
        y=np.array(labels, dtype="float32"),
        tickers=np.array(sample_tickers, dtype=object),
        feature_columns=feature_columns,
        window=window,
        anchor_dates=np.array(anchor_dates),
        label_dates=np.array(label_dates),
    )


def sequence_time_series_split(
    dataset: SequenceDataset, *, test_size: float = 0.2
) -> tuple[SequenceDataset, SequenceDataset]:
    """Return a chronological, per-ticker train/test split of a windowed dataset.

    Samples are already in chronological order within each ticker (as
    ``build_sequence_dataset`` produces them), so a positional cutoff per ticker is used.
    Training samples whose ``label_date`` falls at or after the first test sample's
    ``anchor_date`` are additionally purged — otherwise the tail of the training set
    would be labeled using the same future prices the test set evaluates on.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    train_indices: list[int] = []
    test_indices: list[int] = []

    for ticker_value in pd.unique(dataset.tickers):
        positions = np.flatnonzero(dataset.tickers == ticker_value)
        cutoff = int(len(positions) * (1 - test_size))
        train_candidates = positions[:cutoff]
        if cutoff < len(positions):
            test_start_date = dataset.anchor_dates[positions[cutoff]]
            safe_mask = dataset.label_dates[train_candidates] < test_start_date
            train_candidates = train_candidates[safe_mask]
        train_indices.extend(train_candidates.tolist())
        test_indices.extend(positions[cutoff:].tolist())

    train_index_array = np.array(sorted(train_indices), dtype=int)
    test_index_array = np.array(sorted(test_indices), dtype=int)

    def _subset(indices: np.ndarray) -> SequenceDataset:
        return SequenceDataset(
            X=dataset.X[indices],
            y=dataset.y[indices],
            tickers=dataset.tickers[indices],
            feature_columns=dataset.feature_columns,
            window=dataset.window,
            anchor_dates=dataset.anchor_dates[indices],
            label_dates=dataset.label_dates[indices],
        )

    return _subset(train_index_array), _subset(test_index_array)
