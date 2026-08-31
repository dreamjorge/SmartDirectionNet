# Changelog

All notable changes to SmartDirectionNet will be documented in this file.

## Unreleased

- Added `features.build_sequence_dataset()` and `features.sequence_time_series_split()`
  for windowed (trailing-N-row) supervised dataset construction, respecting ticker
  boundaries and avoiding look-ahead leakage the same way the point-in-time dataset does.
- Added `model.DirectionSequenceClassifier`, a small LSTM alternative to the MLP.
- Added `train.train_sequence_classifier()`, `predict_sequence()`,
  `save_sequence_model()`, and `load_sequence_model()`.
- Added `--model {mlp,lstm}` and `--window` flags to `smartdirectionnet-train`.
- Added `baseline.py`: a LightGBM gradient-boosted-tree baseline (`train_gbm_baseline()`,
  `predict_gbm_baseline()`, `save_gbm_baseline()`, `load_gbm_baseline()`) for comparison
  against the neural networks, wired into the CLI as `--model gbm`.
- Added `--include-macro`/`--macro-series` flags, using SmartAnalyticsInvest's
  `load_stockstreamdb(include_macro=...)` to add FRED macro-economic indicators as
  extra feature columns.
- **Fixed a label-leakage bug** (found via automated code review) in both
  `time_series_split()` and `sequence_time_series_split()`: the naive positional cutoff
  could leave training rows/samples whose *label* was computed from a price at or after
  the first test row's date, letting the model train on the same future prices the test
  set was meant to evaluate on. Both functions now purge those boundary rows/samples.
  `build_direction_dataset()` now carries a `_label_date` column (excluded from
  features) and `SequenceDataset` gained `anchor_dates`/`label_dates` fields to support
  this.
- **Fixed** `predict_sequence()` silently mis-predicting if a `SequenceDataset`'s
  `feature_columns` order didn't match the trained model's; it now reorders matching
  feature sets and raises `ValueError` for a genuinely different feature set.
- **Fixed** `sequence_time_series_split()` raising an uncaught `IndexError` (instead of
  the intended `ValueError` from `train_sequence_classifier`) when every ticker's
  training split was empty, by giving the empty-case index arrays an explicit integer
  dtype.
- **Fixed a look-ahead leakage bug** (found via automated code review): FRED's
  observation date for series like CPI, GDP, and unemployment is the start of the
  reporting period, not the day it was actually published (releases commonly lag two to
  six weeks), so `--include-macro` could leak future information into training. Added
  `--macro-publication-lag-days` (default `30`), threaded through to
  SmartAnalyticsInvest's `load_stockstreamdb(macro_publication_lag_days=...)`, to shift
  observation dates forward by a conservative lag before joining.
- **Fixed** (found via automated code review): the `--macro-publication-lag-days`
  default of `30` was itself not leakage-safe for slower-to-publish series such as
  GDP, whose advance estimate isn't released until roughly 120 days after its
  quarter-start observation date. Raised the default to `130`; documented a smaller
  value (e.g. `45`) as appropriate for callers using only faster monthly series.
- **Fixed a timezone bug** (found via automated code review): `build_direction_dataset()`
  assigned `_label_date` via `.values`, which silently drops timezone info from a
  timezone-aware `date` column. `time_series_split()` then raised `TypeError` comparing
  the resulting timezone-naive `_label_date` against a timezone-aware test-start date,
  breaking the leakage-purge fix entirely for timezone-aware market data.

## 0.1.0 - 2026-08-30

Initial release.

- Added `data_sources`-compatible loading of historical OHLCV/fundamentals/sentiment data
  via SmartAnalyticsInvest's `load_stockstreamdb()`, sourced from StockStreamDB.
- Added `features.build_direction_dataset()` and `features.time_series_split()` for
  leakage-free supervised direction-classification dataset construction.
- Added `model.DirectionClassifier`, a small PyTorch feed-forward network.
- Added `train.train_direction_classifier()`, `predict()`, `save_model()`, and
  `load_model()`.
- Added the `smartdirectionnet-train` end-to-end CLI.
- Added offline pytest coverage, Ruff checks, mypy strict typing, and GitHub Actions CI.
