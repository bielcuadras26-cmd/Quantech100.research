"""Prop firm account simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropFirmRules:
    name: str
    profit_target: float
    max_daily_loss: float
    max_total_loss: float
    risk_per_trade: float
    max_trades_daily: int


@dataclass(frozen=True)
class PropFirmSimulationResult:
    simulations: int
    pass_probability: float
    fail_probability: float
    expected_profit: float
    expected_drawdown: float
    withdrawal_probability: float


PRESETS: dict[str, PropFirmRules] = {
    "FTMO": PropFirmRules("FTMO", 0.10, 0.05, 0.10, 0.005, 8),
    "Topstep": PropFirmRules("Topstep", 0.06, 0.03, 0.06, 0.005, 8),
    "FundingPips": PropFirmRules("FundingPips", 0.08, 0.05, 0.10, 0.005, 8),
    "Alpha Capital": PropFirmRules("Alpha Capital", 0.10, 0.05, 0.10, 0.005, 8),
    "Orion": PropFirmRules("Orion", 0.08, 0.04, 0.08, 0.005, 8),
    "FundingNext": PropFirmRules("FundingNext", 0.10, 0.05, 0.10, 0.005, 8),
}


def simulate_prop_firm(
    trade_r: pd.Series,
    *,
    rules: PropFirmRules,
    initial_capital: float = 100_000.0,
    simulations: int = 10_000,
    seed: int = 42,
) -> PropFirmSimulationResult:
    if trade_r.empty:
        raise ValueError("trade_r must contain at least one trade")
    rng = np.random.default_rng(seed)
    values = trade_r.astype(float).to_numpy()
    pass_events = 0
    fail_events = 0
    withdrawal_events = 0
    profits: list[float] = []
    drawdowns: list[float] = []

    for _ in range(simulations):
        equity = initial_capital
        peak = initial_capital
        max_drawdown = 0.0
        daily_pnl = 0.0
        daily_trades = 0
        passed = False
        failed = False

        for outcome in rng.choice(values, size=max(len(values), 60), replace=True):
            pnl = float(outcome) * rules.risk_per_trade * initial_capital
            equity += pnl
            daily_pnl += pnl
            daily_trades += 1
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

            if daily_pnl <= -rules.max_daily_loss * initial_capital:
                failed = True
                break
            if equity <= initial_capital * (1 - rules.max_total_loss):
                failed = True
                break
            if equity >= initial_capital * (1 + rules.profit_target):
                passed = True
                break
            if daily_trades >= rules.max_trades_daily:
                daily_pnl = 0.0
                daily_trades = 0

        pass_events += int(passed)
        fail_events += int(failed)
        withdrawal_events += int(equity >= initial_capital * 1.03)
        profits.append(equity - initial_capital)
        drawdowns.append(max_drawdown)

    return PropFirmSimulationResult(
        simulations=simulations,
        pass_probability=pass_events / simulations,
        fail_probability=fail_events / simulations,
        expected_profit=float(np.mean(profits)),
        expected_drawdown=float(np.mean(drawdowns)),
        withdrawal_probability=withdrawal_events / simulations,
    )
