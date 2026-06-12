"""Monte Carlo simulation for trade outcome distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    percentile_5: float
    percentile_50: float
    percentile_95: float
    expected_drawdown: float
    worst_drawdown: float
    risk_of_ruin: float
    target_probability: float
    limit_breach_probability: float


def run_monte_carlo(
    trade_r: pd.Series,
    *,
    simulations: int = 10_000,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    ruin_threshold: float = 0.5,
    target_return: float = 0.1,
    max_drawdown_limit: float = 0.1,
    seed: int = 42,
) -> MonteCarloResult:
    if trade_r.empty:
        raise ValueError("trade_r must contain at least one trade")
    rng = np.random.default_rng(seed)
    values = trade_r.astype(float).to_numpy()
    trade_count = len(values)
    final_equities: list[float] = []
    max_drawdowns: list[float] = []
    ruin_events = 0
    target_events = 0
    limit_breaches = 0

    for _ in range(simulations):
        sample = rng.choice(values, size=trade_count, replace=True)
        equity = initial_capital * (1 + np.cumsum(sample * risk_per_trade))
        curve = np.insert(equity, 0, initial_capital)
        peaks = np.maximum.accumulate(curve)
        drawdowns = (curve - peaks) / peaks
        final_equity = float(curve[-1])
        max_drawdown = abs(float(drawdowns.min()))

        final_equities.append(final_equity)
        max_drawdowns.append(max_drawdown)
        ruin_events += int(final_equity <= initial_capital * ruin_threshold)
        target_events += int(final_equity >= initial_capital * (1 + target_return))
        limit_breaches += int(max_drawdown >= max_drawdown_limit)

    return MonteCarloResult(
        simulations=simulations,
        percentile_5=float(np.percentile(final_equities, 5)),
        percentile_50=float(np.percentile(final_equities, 50)),
        percentile_95=float(np.percentile(final_equities, 95)),
        expected_drawdown=float(np.mean(max_drawdowns)),
        worst_drawdown=float(np.max(max_drawdowns)),
        risk_of_ruin=ruin_events / simulations,
        target_probability=target_events / simulations,
        limit_breach_probability=limit_breaches / simulations,
    )
