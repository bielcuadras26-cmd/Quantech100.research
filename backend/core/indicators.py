"""Reusable technical indicators for quantitative research."""

from __future__ import annotations

import pandas as pd


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage returns over the requested number of periods."""

    _validate_window(periods)
    return close.pct_change(periods=periods)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""

    _validate_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""

    _validate_window(span)
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range used by ATR."""

    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range using Wilder-style exponential smoothing."""

    _validate_window(window)
    tr = true_range(high, low, close)
    return _wilder_average(tr, window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder-style smoothing."""

    _validate_window(window)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def macd(
    close: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""

    _validate_window(fast)
    _validate_window(slow)
    _validate_window(signal)
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")

    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        }
    )


def bollinger_bands(
    close: pd.Series,
    *,
    window: int = 20,
    deviations: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands using rolling sample standard deviation."""

    _validate_window(window)
    middle = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = middle + deviations * std
    lower = middle - deviations * std

    return pd.DataFrame(
        {
            "bb_lower": lower,
            "bb_middle": middle,
            "bb_upper": upper,
        }
    )


def z_score(series: pd.Series, window: int) -> pd.Series:
    """Rolling Z-score."""

    _validate_window(window)
    mean = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std


def rolling_volatility(close: pd.Series, window: int, *, annualization: int | None = None) -> pd.Series:
    """Rolling volatility of percentage returns."""

    _validate_window(window)
    volatility = returns(close).rolling(window=window, min_periods=window).std()
    if annualization is not None:
        volatility = volatility * annualization**0.5
    return volatility


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Price momentum over a lookback window."""

    _validate_window(window)
    return close - close.shift(window)


def rolling_correlation(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """Rolling correlation between two aligned series."""

    _validate_window(window)
    return left.rolling(window=window, min_periods=window).corr(right)


def moving_average_slope(series: pd.Series, window: int) -> pd.Series:
    """Difference between current and previous moving average value."""

    average = sma(series, window)
    return average.diff()


def price_to_average_distance_pct(price: pd.Series, average: pd.Series) -> pd.Series:
    """Percentage distance from price to moving average."""

    return (price - average) / average * 100


def price_to_average_distance_atr(
    price: pd.Series,
    average: pd.Series,
    atr_values: pd.Series,
) -> pd.Series:
    """ATR-normalized distance from price to moving average."""

    return (price - average) / atr_values


def rolling_high(high: pd.Series, window: int) -> pd.Series:
    """Rolling highest high."""

    _validate_window(window)
    return high.rolling(window=window, min_periods=window).max()


def rolling_low(low: pd.Series, window: int) -> pd.Series:
    """Rolling lowest low."""

    _validate_window(window)
    return low.rolling(window=window, min_periods=window).min()


def _validate_window(window: int) -> None:
    if window <= 0:
        raise ValueError("window must be positive")


def _wilder_average(values: pd.Series, window: int) -> pd.Series:
    result = pd.Series(index=values.index, dtype="float64")
    if len(values) < window:
        return result

    result.iloc[window - 1] = values.iloc[:window].mean()
    for index in range(window, len(values)):
        previous = result.iloc[index - 1]
        result.iloc[index] = (previous * (window - 1) + values.iloc[index]) / window

    return result
