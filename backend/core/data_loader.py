"""Market data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.core.data_validator import CANONICAL_COLUMNS, normalize_ohlcv_schema, validate_ohlcv


class DataLoadError(ValueError):
    """Raised when a market dataset cannot be loaded safely."""


def load_csv(
    path: str | Path,
    *,
    datetime_column: str = "datetime",
    validate: bool = True,
) -> pd.DataFrame:
    """Load a CSV file into the canonical OHLCV format."""

    source = Path(path)
    if not source.exists():
        raise DataLoadError(f"CSV file does not exist: {source}")

    data = pd.read_csv(source)
    data = normalize_ohlcv_schema(data)

    if datetime_column != "datetime" and datetime_column in data.columns:
        data = data.rename(columns={datetime_column: "datetime"})

    canonical = _canonicalize(data)
    if validate:
        report = validate_ohlcv(canonical)
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise DataLoadError(f"Invalid OHLCV CSV: {messages}")

    return canonical


def load_mt5_export(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a MetaTrader 5 exported CSV file into canonical OHLCV format."""

    source = Path(path)
    if not source.exists():
        raise DataLoadError(f"MT5 export does not exist: {source}")

    data = pd.read_csv(source, sep=None, engine="python")
    data = normalize_ohlcv_schema(data)

    if "<date>" in data.columns and "<time>" in data.columns:
        data["datetime"] = data["<date>"].astype(str) + " " + data["<time>"].astype(str)

    data = data.rename(
        columns={
            "<open>": "open",
            "<high>": "high",
            "<low>": "low",
            "<close>": "close",
            "<tickvol>": "volume",
            "<vol>": "volume",
        }
    )

    canonical = _canonicalize(data)
    if validate:
        report = validate_ohlcv(canonical)
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise DataLoadError(f"Invalid MT5 export: {messages}")

    return canonical


def load_yahoo_finance(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    validate: bool = True,
) -> pd.DataFrame:
    """Load historical data from Yahoo Finance into canonical OHLCV format."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataLoadError("yfinance is required to load Yahoo Finance data.") from exc

    data = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        raise DataLoadError(f"Yahoo Finance returned no rows for symbol: {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data = data.rename(
        columns={
            "Date": "datetime",
            "Datetime": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    canonical = _canonicalize(data)
    if validate:
        report = validate_ohlcv(canonical)
        if not report.is_valid:
            messages = "; ".join(issue.message for issue in report.issues)
            raise DataLoadError(f"Invalid Yahoo Finance data: {messages}")

    return canonical


def _canonicalize(data: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_ohlcv_schema(data)
    missing = [column for column in CANONICAL_COLUMNS if column not in normalized.columns]
    if missing:
        raise DataLoadError(f"Missing required columns: {', '.join(missing)}")

    canonical = normalized.loc[:, CANONICAL_COLUMNS].copy()
    canonical["datetime"] = pd.to_datetime(canonical["datetime"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")

    return canonical.sort_values("datetime").reset_index(drop=True)
