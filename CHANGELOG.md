# Changelog

All notable changes to SmartDirectionNet will be documented in this file.

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
