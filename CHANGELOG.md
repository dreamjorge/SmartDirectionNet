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
