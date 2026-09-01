# Cross-repo integration status

Tracks whether the full pipeline — [StockStreamDB](https://github.com/dreamjorge/StockStreamDB)
collects data, [SmartAnalyticsInvest](https://github.com/dreamjorge/SmartAnalyticsInvest)
cleans and enriches it, this repo builds a dataset and trains a model — actually works
end-to-end against real data, not just against each repo's own (often mocked or
synthetic) test suite. Update this file whenever that's re-verified or a gap is closed.

## Status as of 2026-08-31

| Data source | Status | Notes |
|---|---|---|
| OHLCV (`stock_prices`, via `fetch`) | **Working** | Verified with real Yahoo Finance data (AAPL/MSFT/GOOGL, 5y). Fixed in StockStreamDB#95/#96 — before that, `fetch` didn't persist anything at all. |
| Macro indicators (`macro_indicators`, via `fetch-macro`) | Believed working, not re-verified with this run | Fully implemented and unit-tested (including the publication-lag leakage fix). Not exercised in the 2026-08-31 run — no `FRED_API_KEY` was available in that environment. |
| Fundamentals (`fundamentals`) | **Not implemented upstream** | Table and model exist in StockStreamDB but no fetcher populates it. Tracked as StockStreamDB #7. `--include-fundamentals` will find no data against a real install. |
| Sentiment (`sentiment_analysis`) | **Not implemented upstream** | Same situation as fundamentals. Tracked as StockStreamDB #8, #19, #27. |

## Last verified end-to-end run (2026-08-31)

Real data (not mocked), full pipeline, all three model architectures. See
[issue #13](https://github.com/dreamjorge/SmartDirectionNet/issues/13) for the full
writeup. Baseline results (`--horizon 5`, no tuning — for detecting regressions, not a
quality benchmark):

| Model | Train acc | Test acc |
|---|---|---|
| GBM (`--epochs 100`) | 94.8% | 46.2% (overfit) |
| MLP (`--epochs 30`) | 50.2% | 49.9% |
| LSTM (`--window 20 --epochs 30`) | 55.0% | 55.4% |

## How to re-verify

1. `stockstreamdb fetch <TICKER> 5y` for a few tickers against a scratch database (see
   that project's README — don't point it at the repo's own committed `database.db`
   unless you intend to commit real fetched data, which isn't recommended).
2. Confirm `smartanalyticsinvest.data_sources.load_stockstreamdb()` returns the
   expected shape/columns against that database.
3. `smartdirectionnet-train <db> --output model.pt --model {mlp,lstm,gbm}` for each
   architecture; confirm training completes and the saved model reloads and predicts.
4. Update the table above with the date and results.
