from pathlib import Path

import pandas as pd
import pytest

from backend.core.data_loader import DataLoadError, load_csv, load_mt5_export


def test_load_csv_returns_canonical_sorted_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "market.csv"
    pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-01"],
            "open": [101, 100],
            "high": [102, 101],
            "low": [100, 99],
            "close": [101.5, 100.5],
            "volume": [1100, 1000],
        }
    ).to_csv(csv_path, index=False)

    data = load_csv(csv_path)

    assert list(data.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert data.loc[0, "datetime"] == pd.Timestamp("2026-01-01")


def test_load_csv_raises_for_invalid_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "datetime": ["not-a-date"],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(DataLoadError, match="Invalid OHLCV CSV"):
        load_csv(csv_path)


def test_load_csv_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="does not exist"):
        load_csv(tmp_path / "missing.csv")


def test_load_mt5_export_combines_date_and_time(tmp_path: Path) -> None:
    csv_path = tmp_path / "mt5.csv"
    pd.DataFrame(
        {
            "<DATE>": ["2026.01.01"],
            "<TIME>": ["10:15:00"],
            "<OPEN>": [100],
            "<HIGH>": [101],
            "<LOW>": [99],
            "<CLOSE>": [100.5],
            "<TICKVOL>": [200],
        }
    ).to_csv(csv_path, index=False)

    data = load_mt5_export(csv_path)

    assert data.loc[0, "datetime"] == pd.Timestamp("2026-01-01 10:15:00")
    assert data.loc[0, "volume"] == 200
