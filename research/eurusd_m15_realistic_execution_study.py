"""Corrected EURUSD M15 execution study with realistic cost stress.

This study keeps the full time series intact, calculates indicators on the full sequence,
and applies filters only to entry signals. That avoids inflated results from removing bars
before indicator/backtest calculation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from backend.core.cost_model import CostConfig
from backend.core.data_loader import load_mt5_export
from backend.core.metrics import calculate_metrics
from backend.core.prop_firm_simulator import PRESETS
from backend.strategies.ema_mean_reversion import generate_signals
from research.eurusd_m15_reversion_refinement import add_context, resample_ohlcv
from research.eurusd_m15_walk_forward_funding import (
    build_quarterly_folds,
    simulate_calendar_funding_duration,
)
from research.eurusd_m5_mean_reversion_study import SOURCE, fast_backtest


OUTPUT_DIR = Path("research/results/eurusd_m15_realistic_execution")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = load_mt5_export(SOURCE)
    m15 = add_context(resample_ohlcv(source, "15min"))
    enriched, signals = generate_signals(
        m15,
        ema_window=50,
        atr_window=14,
        distance_atr=2.25,
        exit_distance_atr=0.25,
    )
    signals["short_entry"] = False
    signals["long_entry"] = signals["long_entry"] & (m15["h1_slope"] > 0)

    scenarios = {
        "p95_realistic": CostConfig(
            spread=0.00003,
            commission_per_unit=0.000035,
            slippage=0.00001,
        ),
        "stress": CostConfig(
            spread=0.00005,
            commission_per_unit=0.000035,
            slippage=0.00002,
        ),
    }

    fold_rows = []
    funding_rows = []
    for scenario, costs in scenarios.items():
        trade_books = []
        for fold in build_quarterly_folds(m15, train_months=6, test_months=3):
            mask = (m15["datetime"] >= fold["test_start"]) & (m15["datetime"] < fold["test_end"])
            data = enriched.loc[mask].reset_index(drop=True)
            fold_signals = signals.loc[mask].reset_index(drop=True)
            trades, equity = fast_backtest(
                data,
                fold_signals,
                stop_atr=2.0,
                take_profit_r=2.0,
                max_bars=32,
                costs=costs,
            )
            metrics = calculate_metrics(trades, equity, initial_capital=100_000)
            fold_rows.append(
                {
                    **fold,
                    "scenario": scenario,
                    "trades": len(trades),
                    "expectancy": metrics.net_expectancy,
                    "profit_factor": metrics.net_profit_factor,
                    "win_rate": metrics.win_rate,
                    "total_return": metrics.total_return,
                    "max_drawdown": metrics.max_drawdown,
                }
            )
            if not trades.empty:
                trades = trades.copy()
                trades["fold"] = fold["fold"]
                trade_books.append(trades)

        all_trades = pd.concat(trade_books, ignore_index=True) if trade_books else pd.DataFrame()
        if not all_trades.empty:
            all_trades.to_csv(OUTPUT_DIR / f"{scenario}_walk_forward_trades.csv", index=False)
            for risk in [0.0025, 0.005, 0.0075, 0.01]:
                result = simulate_calendar_funding_duration(
                    all_trades,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    preset="FTMO",
                    risk=risk,
                    simulations=3_000,
                )
                funding_rows.append({"scenario": scenario, **result.__dict__})

    folds = pd.DataFrame(fold_rows)
    funding = pd.DataFrame(funding_rows)
    folds.to_csv(OUTPUT_DIR / "realistic_walk_forward_folds.csv", index=False)
    funding.to_csv(OUTPUT_DIR / "realistic_calendar_funding.csv", index=False)
    write_summary(folds, funding)


def write_summary(folds: pd.DataFrame, funding: pd.DataFrame) -> None:
    lines = [
        "# EURUSD M15 Realistic Execution Study",
        "",
        "Rules:",
        "",
        "- Full M15 timeline preserved",
        "- Indicators calculated before filters",
        "- Long only",
        "- EMA50 M15 mean reversion",
        "- H1 EMA50 slope positive",
        "- Entry distance: 2.25 ATR",
        "- SL: 2 ATR",
        "- TP: 2R",
        "- Max duration: 32 M15 bars",
        "",
        "## Walk-Forward",
        "",
        folds.to_markdown(index=False),
        "",
        "## Calendar-Real FTMO Simulation",
        "",
        funding.to_markdown(index=False),
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
