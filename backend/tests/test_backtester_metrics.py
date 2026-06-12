import pandas as pd

from backend.core.backtester import BacktestConfig, ExitReason, run_backtest
from backend.core.cost_model import CostConfig, Direction
from backend.core.metrics import calculate_metrics


def test_run_backtest_executes_take_profit_with_costs() -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=3, freq="D"),
            "open": [100.0, 100.0, 102.0],
            "high": [100.5, 104.1, 104.0],
            "low": [99.5, 100.5, 101.0],
            "close": [100.0, 103.0, 103.0],
            "atr": [1.0, 1.0, 1.0],
        }
    )
    signals = pd.DataFrame({"long_entry": [True, False, False], "short_entry": [False] * 3})

    result = run_backtest(
        data,
        signals,
        config=BacktestConfig(
            initial_capital=10_000,
            risk_per_trade=0.01,
            default_stop_atr=1,
            default_take_profit_r=2,
        ),
        costs=CostConfig(spread=0.01),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.direction == Direction.LONG
    assert trade.exit_reason == ExitReason.TAKE_PROFIT
    assert trade.gross_r == 2.0
    assert trade.net_r < trade.gross_r


def test_calculate_metrics_from_backtest_trades() -> None:
    trades = pd.DataFrame(
        {
            "gross_result": [100.0, -50.0, 120.0],
            "net_result": [90.0, -60.0, 100.0],
            "gross_r": [1.0, -0.5, 1.2],
            "net_r": [0.9, -0.6, 1.0],
        }
    )
    equity = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=4, freq="D"),
            "equity": [10_000.0, 10_090.0, 10_030.0, 10_130.0],
        }
    )

    metrics = calculate_metrics(trades, equity, initial_capital=10_000.0)

    assert metrics.number_of_trades == 3
    assert metrics.win_rate == 2 / 3
    assert metrics.net_expectancy == (0.9 - 0.6 + 1.0) / 3
    assert metrics.max_losing_streak == 1
