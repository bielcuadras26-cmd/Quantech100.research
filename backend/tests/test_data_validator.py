import pandas as pd

from backend.core.data_validator import normalize_ohlcv_schema, validate_ohlcv


def test_validate_ohlcv_accepts_clean_dataset() -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )

    report = validate_ohlcv(data)

    assert report.is_valid
    assert report.quality_score == 100.0
    assert report.inferred_frequency == "D"


def test_normalize_ohlcv_schema_handles_common_aliases() -> None:
    data = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Open Price": [100],
            "High Price": [101],
            "Low Price": [99],
            "Close Price": [100.5],
            "Vol": [1000],
        }
    )

    normalized = normalize_ohlcv_schema(data)

    assert list(normalized.columns) == ["datetime", "open", "high", "low", "close", "volume"]


def test_validate_ohlcv_reports_missing_columns() -> None:
    data = pd.DataFrame({"datetime": ["2026-01-01"], "close": [100]})

    report = validate_ohlcv(data)

    assert not report.is_valid
    assert report.issues[0].code == "missing_columns"


def test_validate_ohlcv_reports_duplicates_and_chronology() -> None:
    data = pd.DataFrame(
        {
            "datetime": ["2026-01-02", "2026-01-01", "2026-01-01"],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )

    report = validate_ohlcv(data)
    issue_codes = {issue.code for issue in report.issues}

    assert "duplicate_timestamps" in issue_codes
    assert "chronology" in issue_codes


def test_validate_ohlcv_reports_gaps_as_warning() -> None:
    data = pd.DataFrame(
        {
            "datetime": ["2026-01-01", "2026-01-02", "2026-01-04"],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )

    report = validate_ohlcv(data, expected_frequency="D")

    assert report.is_valid
    assert report.warnings[0].code == "time_gaps"


def test_validate_ohlcv_reports_inconsistent_prices() -> None:
    data = pd.DataFrame(
        {
            "datetime": ["2026-01-01"],
            "open": [105],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [1000],
        }
    )

    report = validate_ohlcv(data)

    assert not report.is_valid
    assert report.issues[0].code == "ohlc_consistency"
