"""MT5 DE40 and US500M M15 mean-reversion study with funding simulation."""

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
from research.eurusd_m15_walk_forward_funding import build_quarterly_folds
from research.eurusd_m5_mean_reversion_study import fast_backtest
from research.multi_asset_portfolio_funding import build_asset_calendar, simulate_portfolio_funding


OUTPUT_DIR = Path("research/results/mt5_index_m15_reversion")
INITIAL_CAPITAL = 100_000.0

ASSETS = {
    "DE40": {
        "path": Path("/Users/bielcuadras/Documents/barras baktest/DE40_M5_202409300105_202606121720.csv"),
        "point": 0.01,
        "commission_per_unit": 0.0,
    },
    "US500M": {
        "path": Path("/Users/bielcuadras/Documents/barras baktest/US500M_M5_202412090355_202606121720.csv"),
        "point": 0.01,
        "commission_per_unit": 0.0,
    },
}

BASE_PARAMS = {
    "ema_window": 50,
    "atr_window": 14,
    "distance_atr": 2.25,
    "exit_distance_atr": 0.25,
    "stop_atr": 2.0,
    "take_profit_r": 2.0,
    "max_bars": 32,
}

ROBUSTNESS_GRID = [
    {"distance_atr": distance, "stop_atr": stop, "take_profit_r": tp, "max_bars": max_bars}
    for distance in [1.75, 2.0, 2.25, 2.5]
    for stop in [1.5, 2.0, 2.5]
    for tp in [1.5, 2.0]
    for max_bars in [24, 32, 48]
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_summary_rows = []
    fold_rows = []
    funding_rows = []
    robustness_rows = []
    base_trade_books: dict[str, dict[str, pd.DataFrame]] = {"p95_realistic": {}, "stress": {}}

    for asset, config in ASSETS.items():
        raw = pd.read_csv(config["path"], sep="\t")
        spread_points = raw["<SPREAD>"].astype(float)
        spread_p95 = float(spread_points.quantile(0.95)) * float(config["point"])
        spread_p99 = float(spread_points.quantile(0.99)) * float(config["point"])
        data = resample_ohlcv(load_mt5_export(config["path"]), "15min")
        data = add_context(data)
        data.to_csv(OUTPUT_DIR / f"{asset.lower()}_m15_context.csv", index=False)
        folds = build_quarterly_folds(data, train_months=6, test_months=3)

        scenarios = {
            "p95_realistic": CostConfig(
                spread=spread_p95,
                commission_per_unit=float(config["commission_per_unit"]),
                slippage=spread_p95 * 0.33,
            ),
            "stress": CostConfig(
                spread=max(spread_p99, spread_p95 * 1.5),
                commission_per_unit=float(config["commission_per_unit"]),
                slippage=spread_p95 * 0.75,
            ),
        }

        for scenario, costs in scenarios.items():
            trades, rows = run_walk_forward(asset, data, folds, BASE_PARAMS, costs)
            fold_rows.extend({"scenario": scenario, **row} for row in rows)
            if not trades.empty:
                trades["scenario"] = scenario
                trades.to_csv(OUTPUT_DIR / f"{asset.lower()}_{scenario}_base_trades.csv", index=False)
                base_trade_books[scenario][asset] = trades
                metrics = calculate_metrics(
                    trades,
                    pd.DataFrame(
                        {
                            "datetime": trades["exit_time"],
                            "equity": INITIAL_CAPITAL + trades["net_result"].cumsum(),
                        }
                    ),
                    initial_capital=INITIAL_CAPITAL,
                )
                asset_summary_rows.append(
                    {
                        "asset": asset,
                        "scenario": scenario,
                        "variant": "base",
                        "folds": len(folds),
                        "trades": len(trades),
                        "expectancy": metrics.net_expectancy,
                        "profit_factor": metrics.net_profit_factor,
                        "win_rate": metrics.win_rate,
                        "total_return": metrics.total_return,
                        "max_drawdown": metrics.max_drawdown,
                        "spread_p95": spread_p95,
                        "spread_stress": costs.spread,
                    }
                )
                for risk in [0.0025, 0.005, 0.0075]:
                    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
                    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
                    trades["trade_date"] = trades["entry_time"].dt.date
                    result = simulate_portfolio_funding(
                        {asset: build_asset_calendar(trades)},
                        portfolio=asset,
                        assets=[asset],
                        scenario=scenario,
                        rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                        risk=risk,
                        simulations=2_000,
                    )
                    funding_rows.append({"asset": asset, **result.__dict__})

            for variant_index, partial in enumerate(ROBUSTNESS_GRID, start=1):
                params = {**BASE_PARAMS, **partial}
                variant_trades, variant_rows = run_walk_forward(asset, data, folds, params, costs)
                robustness_rows.append(
                    summarize_variant(asset, scenario, variant_index, params, variant_trades, variant_rows)
                )

    for scenario, books in base_trade_books.items():
        if len(books) < 2:
            continue
        for trades in books.values():
            trades["entry_time"] = pd.to_datetime(trades["entry_time"])
            trades["exit_time"] = pd.to_datetime(trades["exit_time"])
            trades["trade_date"] = trades["entry_time"].dt.date
        asset_days = {asset: build_asset_calendar(trades) for asset, trades in books.items()}
        for risk in [0.0025, 0.005, 0.0075]:
            result = simulate_portfolio_funding(
                asset_days,
                portfolio="DE40+US500M",
                assets=list(books.keys()),
                scenario=scenario,
                rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                risk=risk,
                simulations=2_000,
            )
            funding_rows.append({"asset": "PORTFOLIO", **result.__dict__})

    summary = pd.DataFrame(asset_summary_rows)
    folds = pd.DataFrame(fold_rows)
    funding = pd.DataFrame(funding_rows)
    robustness = pd.DataFrame(robustness_rows)
    summary.to_csv(OUTPUT_DIR / "asset_summary.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "base_walk_forward_folds.csv", index=False)
    funding.to_csv(OUTPUT_DIR / "funding_simulations.csv", index=False)
    robustness.to_csv(OUTPUT_DIR / "robustness_grid.csv", index=False)
    write_summary(summary, folds, funding, robustness)


def run_walk_forward(
    asset: str,
    data: pd.DataFrame,
    folds: list[dict[str, pd.Timestamp | str]],
    params: dict[str, float | int],
    costs: CostConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    enriched, signals = generate_signals(
        data,
        ema_window=int(params["ema_window"]),
        atr_window=int(params["atr_window"]),
        distance_atr=float(params["distance_atr"]),
        exit_distance_atr=float(params["exit_distance_atr"]),
    )
    signals["short_entry"] = False
    signals["long_entry"] = signals["long_entry"] & (data["h1_slope"] > 0)
    trade_books = []
    rows = []
    for fold in folds:
        mask = (data["datetime"] >= fold["test_start"]) & (data["datetime"] < fold["test_end"])
        fold_data = enriched.loc[mask].reset_index(drop=True)
        fold_signals = signals.loc[mask].reset_index(drop=True)
        if len(fold_data) < 100:
            continue
        trades, equity = fast_backtest(
            fold_data,
            fold_signals,
            stop_atr=float(params["stop_atr"]),
            take_profit_r=float(params["take_profit_r"]),
            max_bars=int(params["max_bars"]),
            costs=costs,
        )
        metrics = calculate_metrics(trades, equity, initial_capital=INITIAL_CAPITAL)
        rows.append(
            {
                "asset": asset,
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
            trades["fold"] = fold["fold"]
            trade_books.append(trades)
    return pd.concat(trade_books, ignore_index=True) if trade_books else pd.DataFrame(), rows


def summarize_variant(
    asset: str,
    scenario: str,
    variant_index: int,
    params: dict[str, float | int],
    trades: pd.DataFrame,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    metrics = calculate_metrics(
        trades,
        pd.DataFrame(
            {
                "datetime": trades["exit_time"] if not trades.empty else [],
                "equity": INITIAL_CAPITAL + trades["net_result"].cumsum() if not trades.empty else [],
            }
        ),
        initial_capital=INITIAL_CAPITAL,
    )
    folds = pd.DataFrame(rows)
    return {
        "asset": asset,
        "scenario": scenario,
        "variant": f"v{variant_index}",
        **params,
        "folds": len(rows),
        "positive_folds": int((folds["expectancy"] > 0).sum()) if not folds.empty else 0,
        "trades": len(trades),
        "expectancy": metrics.net_expectancy,
        "profit_factor": metrics.net_profit_factor,
        "win_rate": metrics.win_rate,
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
    }


def write_summary(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    funding: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    lines = [
        "# MT5 Index M15 Reversion Study",
        "",
        "Assets: DE40 and US500M from MT5 M5 exports, resampled to M15.",
        "",
        "Base rules:",
        "",
        "- M15 long-only EMA50 mean reversion",
        "- Entry: close <= EMA50 - 2.25 ATR",
        "- Filter: H1 EMA50 slope positive",
        "- SL 2 ATR, TP 2R, max 32 M15 bars",
        "- Costs from MT5 spread p95 and p99 stress",
        "",
        "## Base Asset Summary",
        "",
    ]
    lines.append(summary.sort_values(["scenario", "expectancy"], ascending=[True, False]).to_markdown(index=False) if not summary.empty else "No base trades.")
    lines.extend(["", "## Base Funding Simulations", ""])
    lines.append(funding.sort_values(["scenario", "portfolio", "risk"]).to_markdown(index=False) if not funding.empty else "No funding results.")
    lines.extend(["", "## Base Walk-Forward Folds", ""])
    lines.append(folds.to_markdown(index=False) if not folds.empty else "No folds.")
    lines.extend(["", "## Top Robustness Variants", ""])
    if robustness.empty:
        lines.append("No robustness results.")
    else:
        top = robustness[
            (robustness["trades"] >= 20)
            & (robustness["positive_folds"] >= (robustness["folds"] - 1))
            & (robustness["expectancy"] > 0)
        ].sort_values(["scenario", "asset", "expectancy"], ascending=[True, True, False])
        lines.append(top.groupby(["scenario", "asset"], as_index=False).head(8).to_markdown(index=False))
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
