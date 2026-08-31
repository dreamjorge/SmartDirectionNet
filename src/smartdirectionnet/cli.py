"""End-to-end CLI: StockStreamDB -> indicators -> features -> training -> saved model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smartanalyticsinvest.data_sources import load_stockstreamdb
from smartanalyticsinvest.errors import SmartAnalyticsInvestError
from smartanalyticsinvest.pipeline import clean_ohlcv, enrich_ohlcv

from smartdirectionnet.baseline import TrainedBaseline, save_gbm_baseline, train_gbm_baseline
from smartdirectionnet.features import (
    build_direction_dataset,
    build_sequence_dataset,
    sequence_time_series_split,
    time_series_split,
)
from smartdirectionnet.train import (
    TrainedModel,
    TrainedSequenceModel,
    save_model,
    save_sequence_model,
    train_direction_classifier,
    train_sequence_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the SmartDirectionNet training CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="smartdirectionnet-train",
        description="Train a direction-classification model from a StockStreamDB database.",
    )
    parser.add_argument("db_path", help="Path to a StockStreamDB SQLite database")
    parser.add_argument("--output", "-o", required=True, help="Path to save the trained model")
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to include (repeatable, default: all tickers in the database)",
    )
    parser.add_argument(
        "--horizon", type=int, default=5, help="Prediction horizon in rows (default: 5)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of each ticker's rows held out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs for mlp/lstm, or boosting rounds for gbm (default: 20)",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "lstm", "gbm"],
        default="mlp",
        help="Model architecture: 'mlp' trains on a single row's indicators (default); "
        "'lstm' trains on a trailing window of rows via an LSTM; "
        "'gbm' trains a LightGBM gradient-boosted tree baseline (not a neural network, "
        "for comparison)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Trailing window length in rows, used only when --model lstm (default: 20)",
    )
    parser.add_argument(
        "--include-fundamentals",
        action="store_true",
        help="Join StockStreamDB fundamentals as extra feature columns",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Join StockStreamDB sentiment scores as extra feature columns",
    )
    parser.add_argument(
        "--include-macro",
        action="store_true",
        help="Join FRED macro indicators (from StockStreamDB's macro_indicators table) "
        "as extra feature columns",
    )
    parser.add_argument(
        "--macro-series",
        action="append",
        dest="macro_series",
        help="FRED series ID to include (repeatable, default: all series in the "
        "database); only used with --include-macro",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end training pipeline from command-line arguments."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    trained: TrainedModel | TrainedSequenceModel | TrainedBaseline
    try:
        raw = load_stockstreamdb(
            args.db_path,
            tickers=args.tickers,
            include_fundamentals=args.include_fundamentals,
            include_sentiment=args.include_sentiment,
            include_macro=args.include_macro,
            macro_series=args.macro_series,
        )
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
        if args.model == "lstm":
            dataset = build_sequence_dataset(enriched, window=args.window, horizon=args.horizon)
            train_set, test_set = sequence_time_series_split(dataset, test_size=args.test_size)
            trained, metrics = train_sequence_classifier(train_set, test_set, epochs=args.epochs)
            train_count, test_count = len(train_set.y), len(test_set.y)
        elif args.model == "gbm":
            frame_dataset = build_direction_dataset(enriched, horizon=args.horizon)
            train_frame, test_frame = time_series_split(frame_dataset, test_size=args.test_size)
            trained, metrics = train_gbm_baseline(
                train_frame, test_frame, num_boost_round=args.epochs
            )
            train_count, test_count = len(train_frame), len(test_frame)
        else:
            frame_dataset = build_direction_dataset(enriched, horizon=args.horizon)
            train_frame, test_frame = time_series_split(frame_dataset, test_size=args.test_size)
            trained, metrics = train_direction_classifier(
                train_frame, test_frame, epochs=args.epochs
            )
            train_count, test_count = len(train_frame), len(test_frame)
    except FileNotFoundError:
        print(f"Error: could not read database file: {args.db_path}", file=sys.stderr)
        return 1
    except (SmartAnalyticsInvestError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    try:
        if isinstance(trained, TrainedSequenceModel):
            save_sequence_model(trained, output_path)
        elif isinstance(trained, TrainedBaseline):
            save_gbm_baseline(trained, output_path)
        else:
            save_model(trained, output_path)
    except OSError as exc:
        print(f"Error: could not write model file: {output_path}: {exc}", file=sys.stderr)
        return 1

    print(f"Trained on {train_count} rows, tested on {test_count} rows")
    print(
        f"Train accuracy: {metrics['train_accuracy']:.3f}  "
        f"Test accuracy: {metrics['test_accuracy']:.3f}"
    )
    print(f"Saved model to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
