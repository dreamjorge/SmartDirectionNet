# SmartDirectionNet

SmartDirectionNet trains a small PyTorch feed-forward neural network to classify whether a
stock's price will be higher N rows ahead ("direction classification"), using the
technical indicators computed by [SmartAnalyticsInvest](https://github.com/dreamjorge/SmartAnalyticsInvest)
as features and historical data collected by
[StockStreamDB](https://github.com/dreamjorge/StockStreamDB) as its data source.

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
published to PyPI), along with PyTorch, pandas, and the dev tooling.

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

## Design notes

- **No look-ahead leakage.** `build_direction_dataset` labels row *i* using the price
  `horizon` rows ahead of it, and drops rows with no future row to label. `time_series_split`
  splits chronologically (never shuffled) and independently per ticker, so every training
  row predates every test row for that ticker.
- **Ticker boundaries are respected.** Labeling never uses another ticker's future rows,
  matching the per-ticker grouping guarantees already provided by SmartAnalyticsInvest's
  pipeline.
- **The model outputs raw logits**, not probabilities — pair it with `BCEWithLogitsLoss`
  during training (already done in `train_direction_classifier`) and apply
  `torch.sigmoid` yourself at inference (already done in `predict`).
- Direction classification, not price prediction: the label is binary (up/down), not a
  regression target. This is deliberately simpler to validate correctly than predicting
  an exact future price.

## Disclaimer

This is an educational/experimental project. Predictions from this model are not
investment advice, and past technical-indicator patterns are not a reliable guide to
future price movements.
