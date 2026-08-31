# SmartDirectionNet

SmartDirectionNet trains a model to classify whether a stock's price will be higher N
rows ahead ("direction classification"), using the technical indicators computed by
[SmartAnalyticsInvest](https://github.com/dreamjorge/SmartAnalyticsInvest) as features
and historical data collected by [StockStreamDB](https://github.com/dreamjorge/StockStreamDB)
as its data source. Three architectures are available: a feed-forward MLP over a single
row's indicator snapshot (the default), an LSTM over a trailing window of rows that
captures the sequence's temporal structure, and a LightGBM gradient-boosted-tree
baseline — included because gradient boosting is the standard, often hard-to-beat
baseline for tabular financial data, and is not a neural network.

This is an experimental, standalone companion project — it exists specifically so that
SmartAnalyticsInvest's core CSV analytics pipeline can stay deterministic and ML-free,
while this repository handles the optional, heavier (PyTorch-based) training workflow.

## How the three projects fit together

```
StockStreamDB            SmartAnalyticsInvest              SmartDirectionNet
(collects OHLCV,     ->  (cleans OHLCV, computes      ->  (builds a labeled dataset,
 fundamentals, and       SMA/RSI/MACD/Bollinger/ATR        trains + saves a PyTorch
 news sentiment into     indicators; reads a               direction-classification
 a local SQLite file)    StockStreamDB database             model)
                         via load_stockstreamdb())
```

## Setup

Use Python 3.14 or newer.

```bash
python3 -m pip install -e '.[dev]'
```

This installs `smartanalyticsinvest` directly from its GitHub repository (it isn't
published to PyPI), along with PyTorch, LightGBM, pandas, and the dev tooling.

## Run tests

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy src/smartdirectionnet
```

Tests are offline and deterministic: SQLite fixtures are built in-memory/on-disk inside
the test itself, and model training uses a fixed random seed.

## End-to-end usage

1. Populate a StockStreamDB SQLite database (see that project's README for how to fetch
   OHLCV, fundamentals, and sentiment data for your tickers).
2. Train and save a model:

```bash
smartdirectionnet-train stockstreamdb.db --output model.pt --horizon 5 --epochs 20
```

Add `--ticker AAPL --ticker MSFT` (repeatable) to restrict to specific tickers, and
`--include-fundamentals`/`--include-sentiment` to also use StockStreamDB's fundamentals
and news-sentiment tables as extra features.

Add `--include-macro` to also use FRED macro-economic indicators (interest rates,
inflation, unemployment, etc., from StockStreamDB's `macro_indicators` table) as extra
features, broadcast to every ticker. Restrict to specific series with `--macro-series`
(repeatable, e.g. `--macro-series FEDFUNDS --macro-series UNRATE`); omit it to include
every series present in the database.

Pass `--model lstm --window 20` to train the LSTM architecture on a trailing 20-row
window instead of the default single-row MLP:

```bash
smartdirectionnet-train stockstreamdb.db --output model.pt --model lstm --window 20 --epochs 20
```

Pass `--model gbm` to train the LightGBM baseline instead (here `--epochs` sets the
number of boosting rounds):

```bash
smartdirectionnet-train stockstreamdb.db --output model.json --model gbm --epochs 100
```

3. Or drive it from Python directly, for more control over indicators and features:

```python
from smartanalyticsinvest.data_sources import load_stockstreamdb
from smartanalyticsinvest.pipeline import clean_ohlcv, enrich_ohlcv

from smartdirectionnet.features import build_direction_dataset, time_series_split
from smartdirectionnet.train import predict, save_model, train_direction_classifier

raw = load_stockstreamdb("stockstreamdb.db", tickers=["AAPL"])
cleaned = clean_ohlcv(raw)
enriched = enrich_ohlcv(
    cleaned,
    sma_windows=(10, 20),
    rsi_windows=(14,),
    ema_windows=(12, 26),
    include_macd=True,
    bollinger_windows=(20,),
    atr_windows=(14,),
)

dataset = build_direction_dataset(enriched, horizon=5)
train_frame, test_frame = time_series_split(dataset, test_size=0.2)

trained, metrics = train_direction_classifier(train_frame, test_frame, epochs=20)
print(metrics)  # {"train_accuracy": ..., "test_accuracy": ...}

save_model(trained, "model.pt")
```

For the LSTM architecture, use the windowed equivalents instead:

```python
from smartdirectionnet.features import build_sequence_dataset, sequence_time_series_split
from smartdirectionnet.train import (
    predict_sequence,
    save_sequence_model,
    train_sequence_classifier,
)

dataset = build_sequence_dataset(enriched, window=20, horizon=5)
train_set, test_set = sequence_time_series_split(dataset, test_size=0.2)

trained, metrics = train_sequence_classifier(train_set, test_set, epochs=20)
save_sequence_model(trained, "model.pt")
```

For the LightGBM baseline, reuse the same point-in-time dataset as the MLP:

```python
from smartdirectionnet.baseline import save_gbm_baseline, train_gbm_baseline

trained, metrics = train_gbm_baseline(train_frame, test_frame, num_boost_round=100)
print(metrics)
save_gbm_baseline(trained, "model.json")
```

## Design notes

- **No look-ahead leakage.** `build_direction_dataset` labels row *i* using the price
  `horizon` rows ahead of it, and drops rows with no future row to label. `time_series_split`
  splits chronologically (never shuffled) and independently per ticker, so every training
  row predates every test row for that ticker — and it additionally purges any training
  row whose *label* was computed from a date at or after the first test row's date (the
  last `horizon`-ish rows before a naive cutoff), since otherwise the tail of the training
  set would be labeled using the same future prices the test set evaluates on.
  `sequence_time_series_split` applies the same purge to windowed samples.
- **Ticker boundaries are respected.** Labeling never uses another ticker's future rows,
  matching the per-ticker grouping guarantees already provided by SmartAnalyticsInvest's
  pipeline.
- **The model outputs raw logits**, not probabilities — pair it with `BCEWithLogitsLoss`
  during training (already done in `train_direction_classifier`) and apply
  `torch.sigmoid` yourself at inference (already done in `predict`).
- Direction classification, not price prediction: the label is binary (up/down), not a
  regression target. This is deliberately simpler to validate correctly than predicting
  an exact future price.
- **Choosing an architecture:** the MLP is simpler, faster to train, and works from very
  little data per ticker. The LSTM captures sequential structure across a trailing window
  and is a better fit once you have enough history per ticker — but is more data-hungry
  and slower to train. The LightGBM baseline (`--model gbm`) is not a neural network at
  all — gradient-boosted trees are the standard, often hard-to-beat baseline for tabular
  financial data, and training one takes seconds. Always train it alongside the MLP/LSTM
  as a sanity check: if a neural network can't beat it, the added complexity isn't
  earning its keep yet.

## Disclaimer

This is an educational/experimental project. Predictions from this model are not
investment advice, and past technical-indicator patterns are not a reliable guide to
future price movements.
