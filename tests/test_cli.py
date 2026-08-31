import math
import sqlite3

from smartdirectionnet.cli import main


def _build_stockstreamdb_fixture(db_path, tickers=("AAPL", "MSFT"), rows=90):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE stock_prices (price_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ticker TEXT, date DATE, open REAL, high REAL, low REAL, close REAL, "
            "adjusted_close REAL, volume INTEGER)"
        )
        records = []
        for ticker_index, ticker in enumerate(tickers):
            for day in range(rows):
                close = 100 + ticker_index * 50 + day * 0.1 + 5 * math.sin(day / 5)
                records.append(
                    (
                        ticker,
                        f"2024-{1 + day // 28:02d}-{1 + day % 28:02d}",
                        close - 0.5,
                        close + 1.0,
                        close - 1.5,
                        close,
                        close - 0.1,
                        1000 + day,
                    )
                )
        connection.executemany(
            "INSERT INTO stock_prices "
            "(ticker, date, open, high, low, close, adjusted_close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )
        connection.execute(
            "CREATE TABLE macro_indicators (macro_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "series_id TEXT, date DATE, value REAL)"
        )
        connection.executemany(
            "INSERT INTO macro_indicators (series_id, date, value) VALUES (?, ?, ?)",
            [("FEDFUNDS", "2023-12-01", 5.25), ("UNRATE", "2023-12-01", 3.7)],
        )


def test_cli_main_trains_and_saves_a_model_end_to_end(tmp_path, capsys):
    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.pt"
    _build_stockstreamdb_fixture(db_path)

    exit_code = main([str(db_path), "--output", str(model_path), "--epochs", "5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert model_path.exists()
    assert "Train accuracy" in captured.out
    assert "Test accuracy" in captured.out
    assert f"Saved model to {model_path}" in captured.out


def test_cli_main_reports_missing_database_without_traceback(tmp_path, capsys):
    exit_code = main([str(tmp_path / "missing.db"), "--output", str(tmp_path / "model.pt")])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error:" in captured.err.lower()
    assert "could not read database file" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_cli_main_filters_by_ticker(tmp_path, capsys):
    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.pt"
    _build_stockstreamdb_fixture(db_path, tickers=("AAPL", "MSFT"))

    exit_code = main(
        [
            str(db_path),
            "--output",
            str(model_path),
            "--ticker",
            "AAPL",
            "--epochs",
            "5",
        ]
    )

    assert exit_code == 0
    assert model_path.exists()


def test_cli_main_trains_lstm_model_end_to_end(tmp_path, capsys):
    from smartdirectionnet.train import load_sequence_model

    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.pt"
    _build_stockstreamdb_fixture(db_path)

    exit_code = main(
        [
            str(db_path),
            "--output",
            str(model_path),
            "--model",
            "lstm",
            "--window",
            "5",
            "--epochs",
            "5",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert model_path.exists()
    assert "Train accuracy" in captured.out
    reloaded = load_sequence_model(model_path)
    assert reloaded.window == 5


def test_cli_main_trains_gbm_baseline_end_to_end(tmp_path, capsys):
    from smartdirectionnet.baseline import load_gbm_baseline

    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.json"
    _build_stockstreamdb_fixture(db_path)

    exit_code = main(
        [
            str(db_path),
            "--output",
            str(model_path),
            "--model",
            "gbm",
            "--epochs",
            "20",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert model_path.exists()
    assert "Train accuracy" in captured.out
    reloaded = load_gbm_baseline(model_path)
    assert reloaded.feature_columns


def test_cli_main_includes_macro_features_end_to_end(tmp_path):
    from smartdirectionnet.baseline import load_gbm_baseline

    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.json"
    _build_stockstreamdb_fixture(db_path)

    exit_code = main(
        [
            str(db_path),
            "--output",
            str(model_path),
            "--model",
            "gbm",
            "--include-macro",
            "--epochs",
            "20",
        ]
    )

    assert exit_code == 0
    assert model_path.exists()
    reloaded = load_gbm_baseline(model_path)
    assert "macro_FEDFUNDS" in reloaded.feature_columns
    assert "macro_UNRATE" in reloaded.feature_columns


def test_cli_main_filters_macro_series(tmp_path):
    from smartdirectionnet.baseline import load_gbm_baseline

    db_path = tmp_path / "stockstream.db"
    model_path = tmp_path / "model.json"
    _build_stockstreamdb_fixture(db_path)

    exit_code = main(
        [
            str(db_path),
            "--output",
            str(model_path),
            "--model",
            "gbm",
            "--include-macro",
            "--macro-series",
            "UNRATE",
            "--epochs",
            "20",
        ]
    )

    assert exit_code == 0
    reloaded = load_gbm_baseline(model_path)
    assert "macro_UNRATE" in reloaded.feature_columns
    assert "macro_FEDFUNDS" not in reloaded.feature_columns
