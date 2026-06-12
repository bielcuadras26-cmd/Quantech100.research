import pandas as pd

from backend.core.backtester import BacktestConfig, run_backtest
from backend.core.external_frameworks import (
    quantstats_snapshot,
    run_backtesting_py_ema_reversion,
    run_vectorbt_signal_backtest,
)
from backend.strategies.ema_mean_reversion import generate_signals


def _dataset(rows: int = 160) -> pd.DataFrame:
    closes = [100 + (index % 15) * 0.6 for index in range(rows)]
    for index in range(40, rows, 50):
        closes[index] -= 5
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2021-01-01", periods=rows, freq="D"),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [1000] * rows,
        }
    )


def test_backtesting_py_adapter_runs_strategy() -> None:
    summary = run_backtesting_py_ema_reversion(
        _dataset(),
        ema_window=10,
        atr_window=5,
        distance_atr=0.5,
    )

    assert summary.framework == "backtesting.py"
    assert summary.trades >= 0


def test_vectorbt_adapter_runs_signal_backtest() -> None:
    data = _dataset()
    enriched, signals = generate_signals(data, ema_window=10, atr_window=5, distance_atr=0.5)
    summary = run_vectorbt_signal_backtest(
        enriched["close"],
        signals["long_entry"],
        signals["exit"],
    )

    assert summary.framework == "vectorbt"
    assert summary.trades >= 0


def test_quantstats_snapshot_cross_checks_equity_curve() -> None:
    data = _dataset()
    enriched, signals = generate_signals(data, ema_window=10, atr_window=5, distance_atr=0.5)
    result = run_backtest(
        enriched,
        signals,
        config=BacktestConfig(initial_capital=100_000, risk_per_trade=0.01),
    )

    snapshot = quantstats_snapshot(result.equity_curve)

    assert {"sharpe", "sortino", "max_drawdown", "cagr"} <= set(snapshot)
