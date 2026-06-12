"""Performance and risk metrics for backtest outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricsSnapshot:
    number_of_trades: int
    win_rate: float
    gross_profit_factor: float
    net_profit_factor: float
    gross_expectancy: float
    net_expectancy: float
    total_return: float
    max_drawdown: float
    relative_drawdown: float
    max_losing_streak: int
    sharpe_ratio: float
    sortino_ratio: float
    recovery_factor: float


def calculate_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    initial_capital: float,
) -> MetricsSnapshot:
    if trades.empty:
        return MetricsSnapshot(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    gross = trades["gross_result"].astype(float)
    net = trades["net_result"].astype(float)
    net_r = trades["net_r"].astype(float)
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    drawdowns = calculate_drawdown(equity)
    total_return = (float(equity.iloc[-1]) - initial_capital) / initial_capital
    max_drawdown = float(drawdowns.min())

    return MetricsSnapshot(
        number_of_trades=len(trades),
        win_rate=float((net > 0).mean()),
        gross_profit_factor=_profit_factor(gross),
        net_profit_factor=_profit_factor(net),
        gross_expectancy=float(trades["gross_r"].mean()),
        net_expectancy=float(net_r.mean()),
        total_return=total_return,
        max_drawdown=max_drawdown,
        relative_drawdown=abs(max_drawdown),
        max_losing_streak=_max_losing_streak(net),
        sharpe_ratio=_sharpe(returns),
        sortino_ratio=_sortino(returns),
        recovery_factor=abs(total_return / max_drawdown) if max_drawdown < 0 else 0.0,
    )


def calculate_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def results_by_period(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "net_result", "net_r", "trades"])
    indexed = trades.copy()
    indexed["exit_time"] = pd.to_datetime(indexed["exit_time"])
    grouped = indexed.groupby(indexed["exit_time"].dt.to_period(frequency))
    return grouped.agg(net_result=("net_result", "sum"), net_r=("net_r", "sum"), trades=("net_r", "size"))


def _profit_factor(results: pd.Series) -> float:
    wins = float(results[results > 0].sum())
    losses = abs(float(results[results < 0].sum()))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _max_losing_streak(results: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in results:
        if value < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def _sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if returns.empty or downside.empty or downside.std() == 0:
        return 0.0
    return float((returns.mean() / downside.std()) * np.sqrt(252))
