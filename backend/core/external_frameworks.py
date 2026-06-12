"""Adapters around external quant frameworks.

These adapters are intentionally used for fast research and cross-checking, not as hidden
replacements for the transparent QuantTech100 core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backend.core.indicators import atr, ema, price_to_average_distance_atr


@dataclass(frozen=True)
class ExternalBacktestSummary:
    framework: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: int
    raw_stats: dict[str, Any]


def run_backtesting_py_ema_reversion(
    data: pd.DataFrame,
    *,
    ema_window: int = 100,
    atr_window: int = 14,
    distance_atr: float = 2.0,
    cash: float = 100_000.0,
    commission: float = 0.0005,
) -> ExternalBacktestSummary:
    """Run an EMA/ATR mean-reversion strategy using Backtesting.py."""

    from backtesting import Backtest, Strategy

    prepared = _prepare_backtesting_py_data(data)

    class EmaAtrMeanReversion(Strategy):  # type: ignore[misc]
        def init(self) -> None:
            close = pd.Series(self.data.Close, dtype="float64")
            high = pd.Series(self.data.High, dtype="float64")
            low = pd.Series(self.data.Low, dtype="float64")
            average = ema(close, ema_window)
            atr_values = atr(high, low, close, atr_window)
            distance = price_to_average_distance_atr(close, average, atr_values)
            self.distance = self.I(lambda: distance.to_numpy())

        def next(self) -> None:
            distance = self.distance[-1]
            if pd.isna(distance):
                return
            if self.position:
                if abs(distance) <= 0.25:
                    self.position.close()
                return
            if distance <= -distance_atr:
                self.buy()
            elif distance >= distance_atr:
                self.sell()

    backtest = Backtest(
        prepared,
        EmaAtrMeanReversion,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )
    stats = backtest.run()
    return ExternalBacktestSummary(
        framework="backtesting.py",
        total_return=float(stats.get("Return [%]", 0.0)) / 100,
        max_drawdown=float(stats.get("Max. Drawdown [%]", 0.0)) / 100,
        sharpe_ratio=float(stats.get("Sharpe Ratio", 0.0)),
        trades=int(stats.get("# Trades", 0)),
        raw_stats={str(key): _to_builtin(value) for key, value in stats.items()},
    )


def run_vectorbt_signal_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    *,
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
) -> ExternalBacktestSummary:
    """Run a vectorbt signal backtest for fast cross-checking."""

    import vectorbt as vbt

    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        freq="1D",
    )
    total_return = float(portfolio.total_return())
    max_drawdown = float(portfolio.max_drawdown())
    sharpe = float(portfolio.sharpe_ratio() or 0.0)
    trades = int(portfolio.trades.count())
    return ExternalBacktestSummary(
        framework="vectorbt",
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        trades=trades,
        raw_stats={
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "trades": trades,
        },
    )


def quantstats_snapshot(equity_curve: pd.DataFrame) -> dict[str, float]:
    """Calculate selected QuantStats metrics from a QuantTech100 equity curve."""

    import quantstats as qs

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "cagr": 0.0}
    return {
        "sharpe": float(qs.stats.sharpe(returns)),
        "sortino": float(qs.stats.sortino(returns)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
        "cagr": float(qs.stats.cagr(returns)),
    }


def _prepare_backtesting_py_data(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    prepared["datetime"] = pd.to_datetime(prepared["datetime"])
    prepared = prepared.set_index("datetime")
    prepared = prepared.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return prepared.loc[:, ["Open", "High", "Low", "Close", "Volume"]]


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
