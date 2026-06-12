import pandas as pd

from backend.core.validation import (
    build_walk_forward_splits,
    run_out_of_sample_validation,
    run_walk_forward_validation,
    train_test_split,
)


def _dataset(rows: int = 180) -> pd.DataFrame:
    closes = [100.0 + (index % 12) * 0.5 for index in range(rows)]
    for index in range(30, rows, 45):
        closes[index] -= 5.0
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1000] * rows,
        }
    )


def test_train_test_split_is_chronological() -> None:
    train, test = train_test_split(_dataset(10), train_ratio=0.6)

    assert len(train) == 6
    assert len(test) == 4
    assert train.iloc[-1]["datetime"] < test.iloc[0]["datetime"]


def test_build_walk_forward_splits() -> None:
    splits = build_walk_forward_splits(_dataset(100), train_size=40, test_size=20)

    assert len(splits) == 3
    assert splits[0].train_end < splits[0].test_start


def test_run_out_of_sample_validation_selects_parameters() -> None:
    result = run_out_of_sample_validation(
        _dataset(),
        parameter_grid={
            "ema_window": [10, 20],
            "atr_window": [5],
            "distance_atr": [0.5, 1.0],
        },
        train_ratio=0.7,
    )

    assert result.split.name == "out_of_sample"
    assert result.selected_parameters["ema_window"] in {10, 20}


def test_run_walk_forward_validation_returns_stability_score() -> None:
    summary = run_walk_forward_validation(
        _dataset(),
        parameter_grid={
            "ema_window": [10],
            "atr_window": [5],
            "distance_atr": [0.5, 1.0],
        },
        train_size=80,
        test_size=30,
    )

    assert summary.aggregate_test_metrics["folds"] > 0
    assert 0 <= summary.stability_score <= 1
