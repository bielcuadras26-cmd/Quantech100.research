"""Multi-asset mean-reversion comparison for EURUSD, XAUUSD, and US100."""

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
from research.eurusd_m5_mean_reversion_study import fast_backtest


OUTPUT_DIR = Path("research/results/multi_asset_reversion")

ASSETS = {
    "EURUSD": {
        "path": Path("/Users/bielcuadras/Documents/barras baktest/EURUSD_M5_202501070115_202605111510.csv"),
        "source_tf": "M5",
        "target_rule": "15min",
        "point": 0.00001,
        "commission_per_unit": 0.000035,
    },
    "XAUUSD": {
        "path": Path("/Users/bielcuadras/Documents/barras baktest/XAUUSD_M5_202412091410_202605111510.csv"),
        "source_tf": "M5",
        "target_rule": "15min",
        "point": 0.01,
        "commission_per_unit": 0.035,
    },
    "US100": {
        "path": Path("/Users/bielcuadras/Documents/barras baktest/US100_M15_202201201430_202604281030.csv"),
        "source_tf": "M15",
        "target_rule": None,
        "point": 0.01,
        "commission_per_unit": 0.0,
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fold_rows = []
    funding_rows = []

    for asset, config in ASSETS.items():
        raw = pd.read_csv(config["path"], sep="\t")
        spread_points = raw["<SPREAD>"].astype(float)
        data = load_mt5_export(config["path"])
        if config["target_rule"] is not None:
            data = resample_ohlcv(data, str(config["target_rule"]))
        data = add_context(data)
        data.to_csv(OUTPUT_DIR / f"{asset.lower()}_m15_context.csv", index=False)
        spread_cost = float(spread_points.quantile(0.95)) * float(config["point"])
        stress_spread_cost = max(float(spread_points.quantile(0.99)), float(spread_points.quantile(0.95)) * 1.5) * float(config["point"])

        scenarios = {
            "p95_realistic": CostConfig(
                spread=spread_cost,
                commission_per_unit=float(config["commission_per_unit"]),
                slippage=spread_cost * 0.33,
            ),
            "stress": CostConfig(
                spread=stress_spread_cost,
                commission_per_unit=float(config["commission_per_unit"]),
                slippage=spread_cost * 0.75,
            ),
        }

        enriched, signals = generate_signals(
            data,
            ema_window=50,
            atr_window=14,
            distance_atr=2.25,
            exit_distance_atr=0.25,
        )
        signals["short_entry"] = False
        signals["long_entry"] = signals["long_entry"] & (data["h1_slope"] > 0)

        for scenario, costs in scenarios.items():
            trade_books = []
            folds = build_quarterly_folds(data, train_months=6, test_months=3)
            for fold in folds:
                mask = (data["datetime"] >= fold["test_start"]) & (data["datetime"] < fold["test_end"])
                fold_data = enriched.loc[mask].reset_index(drop=True)
                fold_signals = signals.loc[mask].reset_index(drop=True)
                if len(fold_data) < 100:
                    continue
                trades, equity = fast_backtest(
                    fold_data,
                    fold_signals,
                    stop_atr=2.0,
                    take_profit_r=2.0,
                    max_bars=32,
                    costs=costs,
                )
                metrics = calculate_metrics(trades, equity, initial_capital=100_000)
                fold_rows.append(
                    {
                        "asset": asset,
                        "scenario": scenario,
                        **fold,
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
                    trades["asset"] = asset
                    trades["scenario"] = scenario
                    trades["fold"] = fold["fold"]
                    trade_books.append(trades)

            all_trades = pd.concat(trade_books, ignore_index=True) if trade_books else pd.DataFrame()
            if not all_trades.empty:
                all_trades.to_csv(OUTPUT_DIR / f"{asset.lower()}_{scenario}_trades.csv", index=False)
                equity = pd.DataFrame(
                    {
                        "datetime": all_trades["exit_time"],
                        "equity": 100_000 + all_trades["net_result"].cumsum(),
                    }
                )
                combined = calculate_metrics(all_trades, equity, initial_capital=100_000)
                summary_rows.append(
                    {
                        "asset": asset,
                        "scenario": scenario,
                        "folds": len(folds),
                        "trades": len(all_trades),
                        "expectancy": combined.net_expectancy,
                        "profit_factor": combined.net_profit_factor,
                        "win_rate": combined.win_rate,
                        "total_return": combined.total_return,
                        "max_drawdown": combined.max_drawdown,
                        "spread_p95": spread_cost,
                        "spread_stress": stress_spread_cost,
                    }
                )
                for risk in [0.0025, 0.005, 0.0075]:
                    result = simulate_calendar_funding_duration(
                        all_trades,
                        rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                        preset="FTMO",
                        risk=risk,
                        simulations=3_000,
                    )
                    funding_rows.append({"asset": asset, "scenario": scenario, **result.__dict__})

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    funding = pd.DataFrame(funding_rows)
    summary.to_csv(OUTPUT_DIR / "asset_summary.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "asset_walk_forward_folds.csv", index=False)
    funding.to_csv(OUTPUT_DIR / "asset_funding_simulations.csv", index=False)
    write_summary(summary, folds, funding)


def write_summary(summary: pd.DataFrame, folds: pd.DataFrame, funding: pd.DataFrame) -> None:
    lines = [
        "# Multi-Asset Mean Reversion Comparison",
        "",
        "Rules:",
        "",
        "- M15 long-only mean reversion",
        "- EMA50 distance 2.25 ATR",
        "- H1 EMA50 slope positive",
        "- SL 2 ATR, TP 2R, max 32 M15 bars",
        "- Calendar-real FTMO simulation",
        "",
        "## Combined Asset Summary",
        "",
        summary.sort_values(["scenario", "expectancy"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Walk-Forward Folds",
        "",
        folds.to_markdown(index=False),
        "",
        "## FTMO Calendar-Real Simulations",
        "",
        funding.sort_values(["asset", "scenario", "risk"]).to_markdown(index=False),
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
