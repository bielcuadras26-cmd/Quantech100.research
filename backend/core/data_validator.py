"""Dataset validation for canonical OHLCV market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pandas as pd


CANONICAL_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataQualityIssue:
    """A single data-quality finding."""

    code: str
    message: str
    severity: str = "error"
    count: int | None = None


@dataclass(frozen=True)
class DataValidationReport:
    """Validation result for one market dataset."""

    rows: int
    issues: tuple[DataQualityIssue, ...] = field(default_factory=tuple)
    warnings: tuple[DataQualityIssue, ...] = field(default_factory=tuple)
    inferred_frequency: str | None = None

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def quality_score(self) -> float:
        if self.rows == 0:
            return 0.0

        weighted_count = 0.0
        for issue in self.issues:
            weighted_count += issue.count if issue.count is not None else self.rows
        for warning in self.warnings:
            weighted_count += max(1, warning.count or 1) * 0.25

        return max(0.0, round(100.0 * (1.0 - weighted_count / max(self.rows, 1)), 2))


def normalize_ohlcv_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with canonical OHLCV column names."""

    normalized = data.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    aliases = {
        "date": "datetime",
        "time": "datetime",
        "timestamp": "datetime",
        "open price": "open",
        "high price": "high",
        "low price": "low",
        "close price": "close",
        "tickvol": "volume",
        "tick_volume": "volume",
        "vol": "volume",
    }
    normalized = normalized.rename(columns=aliases)

    return normalized


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    expected_frequency: str | None = None,
    allow_zero_volume: bool = True,
) -> DataValidationReport:
    """Validate canonical OHLCV data without mutating the source dataframe."""

    issues: list[DataQualityIssue] = []
    warnings: list[DataQualityIssue] = []
    normalized = normalize_ohlcv_schema(data)

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in normalized.columns]
    if missing_columns:
        issues.append(
            DataQualityIssue(
                code="missing_columns",
                message=f"Missing required columns: {', '.join(missing_columns)}",
                count=len(missing_columns),
            )
        )
        return DataValidationReport(rows=len(normalized), issues=tuple(issues))

    checked = normalized.loc[:, CANONICAL_COLUMNS].copy()
    checked["datetime"] = pd.to_datetime(checked["datetime"], errors="coerce", utc=False)

    invalid_datetimes = int(checked["datetime"].isna().sum())
    if invalid_datetimes:
        issues.append(
            DataQualityIssue(
                code="invalid_datetime",
                message="Rows contain invalid or missing datetimes.",
                count=invalid_datetimes,
            )
        )

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")

    missing_values = int(checked.isna().sum().sum())
    if missing_values:
        issues.append(
            DataQualityIssue(
                code="missing_values",
                message="Rows contain missing or non-numeric OHLCV values.",
                count=missing_values,
            )
        )

    duplicated_timestamps = int(checked["datetime"].duplicated(keep=False).sum())
    if duplicated_timestamps:
        issues.append(
            DataQualityIssue(
                code="duplicate_timestamps",
                message="Dataset contains duplicate timestamps.",
                count=duplicated_timestamps,
            )
        )

    valid_datetimes = checked["datetime"].dropna()
    if len(valid_datetimes) > 1 and not valid_datetimes.is_monotonic_increasing:
        issues.append(
            DataQualityIssue(
                code="chronology",
                message="Datetime column must be strictly chronological.",
            )
        )

    price_columns = ["open", "high", "low", "close"]
    non_positive_prices = int((checked[price_columns] <= 0).sum().sum())
    if non_positive_prices:
        issues.append(
            DataQualityIssue(
                code="non_positive_prices",
                message="OHLC prices must be positive.",
                count=non_positive_prices,
            )
        )

    invalid_high_low = int((checked["high"] < checked["low"]).sum())
    invalid_open_close = int(
        (
            (checked["open"] > checked["high"])
            | (checked["open"] < checked["low"])
            | (checked["close"] > checked["high"])
            | (checked["close"] < checked["low"])
        ).sum()
    )
    if invalid_high_low or invalid_open_close:
        issues.append(
            DataQualityIssue(
                code="ohlc_consistency",
                message="OHLC relationships are inconsistent.",
                count=invalid_high_low + invalid_open_close,
            )
        )

    if not allow_zero_volume:
        non_positive_volume = int((checked["volume"] <= 0).sum())
        if non_positive_volume:
            issues.append(
                DataQualityIssue(
                    code="non_positive_volume",
                    message="Volume must be positive for this dataset.",
                    count=non_positive_volume,
                )
            )

    inferred_frequency = _infer_frequency(valid_datetimes)
    gap_count = _count_gaps(valid_datetimes, expected_frequency or inferred_frequency)
    if gap_count:
        warnings.append(
            DataQualityIssue(
                code="time_gaps",
                message="Dataset contains possible time gaps.",
                severity="warning",
                count=gap_count,
            )
        )

    return DataValidationReport(
        rows=len(checked),
        issues=tuple(issues),
        warnings=tuple(warnings),
        inferred_frequency=inferred_frequency,
    )


def _infer_frequency(datetimes: pd.Series) -> str | None:
    if len(datetimes) < 3:
        return None

    try:
        return cast(str | None, pd.infer_freq(datetimes))
    except ValueError:
        return None


def _count_gaps(datetimes: pd.Series, frequency: str | None) -> int:
    if frequency is None or len(datetimes) < 3:
        return 0

    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
        expected_delta = pd.Timedelta(offset.nanos, unit="ns")
    except ValueError:
        return 0

    deltas = datetimes.sort_values().diff().dropna()
    return int((deltas > expected_delta).sum())
