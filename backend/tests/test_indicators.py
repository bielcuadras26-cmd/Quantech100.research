import math

import pandas as pd
import pytest

from backend.core.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    momentum,
    moving_average_slope,
    price_to_average_distance_atr,
    price_to_average_distance_pct,
    returns,
    rolling_correlation,
    rolling_high,
    rolling_low,
    rolling_volatility,
    rsi,
    sma,
    true_range,
    z_score,
)


def test_sma_preserves_warmup_period() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = sma(values, 3)

    assert math.isnan(result.iloc[0])
    assert result.iloc[2] == 2.0
    assert result.iloc[3] == 3.0


def test_ema_matches_pandas_reference() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = ema(values, 3)
    expected = values.ewm(span=3, adjust=False, min_periods=3).mean()

    pd.testing.assert_series_equal(result, expected)


def test_true_range_uses_previous_close() -> None:
    high = pd.Series([10.0, 13.0])
    low = pd.Series([8.0, 11.0])
    close = pd.Series([9.0, 12.0])

    result = true_range(high, low, close)

    assert result.tolist() == [2.0, 4.0]


def test_atr_returns_wilder_smoothed_average() -> None:
    high = pd.Series([10.0, 12.0, 13.0, 14.0])
    low = pd.Series([8.0, 9.0, 10.0, 11.0])
    close = pd.Series([9.0, 10.0, 12.0, 13.0])

    result = atr(high, low, close, window=3)

    assert math.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.6666666667)
    assert result.iloc[3] == pytest.approx(2.7777777778)


def test_rsi_bounds_output_between_zero_and_one_hundred() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 4.0, 6.0])

    result = rsi(close, window=3).dropna()

    assert ((result >= 0) & (result <= 100)).all()


def test_macd_returns_line_signal_and_histogram() -> None:
    close = pd.Series([float(value) for value in range(1, 60)])

    result = macd(close)

    assert list(result.columns) == ["macd", "macd_signal", "macd_histogram"]
    assert result["macd_histogram"].dropna().iloc[-1] == pytest.approx(
        result["macd"].dropna().iloc[-1] - result["macd_signal"].dropna().iloc[-1]
    )


def test_bollinger_bands_use_rolling_std() -> None:
    close = pd.Series([1.0, 2.0, 3.0])

    result = bollinger_bands(close, window=3, deviations=2.0)

    assert result.loc[2, "bb_middle"] == 2.0
    assert result.loc[2, "bb_upper"] == 4.0
    assert result.loc[2, "bb_lower"] == 0.0


def test_z_score_and_rolling_volatility() -> None:
    close = pd.Series([100.0, 101.0, 103.0, 102.0])

    assert z_score(close, 3).iloc[2] == pytest.approx(1.0910894512)
    assert not math.isnan(rolling_volatility(close, 2).iloc[2])


def test_momentum_returns_and_correlation() -> None:
    left = pd.Series([1.0, 2.0, 3.0, 4.0])
    right = pd.Series([2.0, 4.0, 6.0, 8.0])

    assert returns(left).iloc[1] == 1.0
    assert momentum(left, 2).iloc[3] == 2.0
    assert rolling_correlation(left, right, 3).iloc[3] == pytest.approx(1.0)


def test_distance_and_rolling_extremes() -> None:
    price = pd.Series([100.0, 105.0, 110.0])
    average = pd.Series([100.0, 100.0, 100.0])
    atr_values = pd.Series([2.0, 2.5, 5.0])

    assert price_to_average_distance_pct(price, average).iloc[2] == 10.0
    assert price_to_average_distance_atr(price, average, atr_values).iloc[2] == 2.0
    assert rolling_high(price, 2).iloc[2] == 110.0
    assert rolling_low(price, 2).iloc[2] == 105.0


def test_moving_average_slope_and_window_validation() -> None:
    values = pd.Series([1.0, 2.0, 4.0, 8.0])

    assert moving_average_slope(values, 2).iloc[3] == 3.0
    with pytest.raises(ValueError, match="window must be positive"):
        sma(values, 0)
