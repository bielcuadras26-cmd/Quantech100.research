"""Yahoo Finance H1 index mean-reversion proxy study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.cost_model import CostConfig
from backend.core.data_loader import load_yahoo_finance
from backend.core.metrics import calculate_metrics
from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from backend.strategies.ema_mean_reversion import generate_signals
from research.eurusd_m15_reversion_refinement import add_context, resample_ohlcv
from research.eurusd_m15_walk_forward_funding import build_quarterly_folds
from research.eurusd_m5_mean_reversion_study import fast_backtest


OUTPUT_DIR = Path("research/results/yahoo_index_h1_reversion")
INITIAL_CAPITAL = 100_000.0
MAX_CALENDAR_DAYS = 90
SIMULATIONS = 2_000

ASSETS = {
    "NASDAQ_NQ": {
        "symbol": "NQ=F",
        "start": "2024-06-15",
        "costs": {
            "conservative": CostConfig(spread=2.0, slippage=1.0),
            "stress": CostConfig(spread=4.0, slippage=2.0),
        },
    },
    "SP500_ES": {
        "symbol": "ES=F",
        "start": "2024-06-15",
        "costs": {
            "conservative": CostConfig(spread=0.50, slippage=0.25),
            "stress": CostConfig(spread=1.00, slippage=0.50),
        },
    },
    "DAX": {
        "symbol": "^GDAXI",
        "start": "2024-06-15",
        "costs": {
            "conservative": CostConfig(spread=2.0, slippage=1.0),
            "stress": CostConfig(spread=4.0, slippage=2.0),
        },
    },
}


@dataclass(frozen=True)
class PortfolioFundingResult:
    scenario: str
    risk: float
    simulations: int
    pass_probability: float
    fail_probability: float
    unresolved_probability: float
    average_calendar_days_to_pass: float | None
    median_calendar_days_to_pass: float | None
    expected_profit: float
    average_trades_used: float


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    fold_rows = []
    scenario_trades: dict[str, list[pd.DataFrame]] = {"conservative": [], "stress": []}

    for asset, config in ASSETS.items():
        data = load_yahoo_finance(
            str(config["symbol"]),
            start=str(config["start"]),
            interval="1h",
        )
        data["datetime"] = pd.to_datetime(data["datetime"], utc=True).dt.tz_convert(None)
        data = data.dropna().reset_index(drop=True)
        data.to_csv(OUTPUT_DIR / f"{asset.lower()}_h1_yahoo.csv", index=False)
        data = add_h4_context(add_context(data))

        enriched, signals = generate_signals(
            data,
            ema_window=50,
            atr_window=14,
            distance_atr=2.25,
            exit_distance_atr=0.25,
        )
        signals["short_entry"] = False
        signals["long_entry"] = signals["long_entry"] & (data["h4_slope"] > 0)

        folds = build_quarterly_folds(data, train_months=6, test_months=3)
        for scenario, costs in dict(config["costs"]).items():
            trade_books = []
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
                    max_bars=16,
                    costs=costs,
                )
                metrics = calculate_metrics(trades, equity, initial_capital=INITIAL_CAPITAL)
                fold_rows.append(
                    {
                        "asset": asset,
                        "symbol": config["symbol"],
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
                    trades["symbol"] = config["symbol"]
                    trades["scenario"] = scenario
                    trades["fold"] = fold["fold"]
                    trade_books.append(trades)

            all_trades = pd.concat(trade_books, ignore_index=True) if trade_books else pd.DataFrame()
            if not all_trades.empty:
                all_trades.to_csv(OUTPUT_DIR / f"{asset.lower()}_{scenario}_trades.csv", index=False)
                scenario_trades[scenario].append(all_trades)
                equity = pd.DataFrame(
                    {
                        "datetime": all_trades["exit_time"],
                        "equity": INITIAL_CAPITAL + all_trades["net_result"].cumsum(),
                    }
                )
                metrics = calculate_metrics(all_trades, equity, initial_capital=INITIAL_CAPITAL)
                summary_rows.append(
                    {
                        "asset": asset,
                        "symbol": config["symbol"],
                        "scenario": scenario,
                        "folds": len(folds),
                        "trades": len(all_trades),
                        "expectancy": metrics.net_expectancy,
                        "profit_factor": metrics.net_profit_factor,
                        "win_rate": metrics.win_rate,
                        "total_return": metrics.total_return,
                        "max_drawdown": metrics.max_drawdown,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    funding_rows = []
    for scenario, books in scenario_trades.items():
        if not books:
            continue
        trades = pd.concat(books, ignore_index=True)
        trades.to_csv(OUTPUT_DIR / f"portfolio_{scenario}_trades.csv", index=False)
        for risk in [0.0025, 0.005, 0.0075]:
            funding_rows.append(
                simulate_portfolio_funding(
                    trades,
                    scenario=scenario,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    risk=risk,
                    simulations=SIMULATIONS,
                ).__dict__
            )
    funding = pd.DataFrame(funding_rows)
    summary.to_csv(OUTPUT_DIR / "asset_summary.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "walk_forward_folds.csv", index=False)
    funding.to_csv(OUTPUT_DIR / "portfolio_funding.csv", index=False)
    write_summary(summary, folds, funding)


def simulate_portfolio_funding(
    trades: pd.DataFrame,
    *,
    scenario: str,
    rules: PropFirmRules,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> PortfolioFundingResult:
    rng = np.random.default_rng(seed)
    prepared = trades.copy()
    prepared["entry_time"] = pd.to_datetime(prepared["entry_time"])
    prepared["trade_date"] = prepared["entry_time"].dt.date
    grouped = {date: group.copy() for date, group in prepared.groupby("trade_date", sort=True)}
    start = pd.Timestamp(prepared["entry_time"].min()).normalize()
    end = pd.Timestamp(prepared["entry_time"].max()).normalize()
    days = []
    for date in pd.date_range(start, end, freq="D"):
        group = grouped.get(date.date())
        if group is None or group.empty:
            days.append(np.empty((0, 2), dtype=float))
        else:
            days.append(group.sort_values(["entry_time", "exit_time"])[["net_r", "mae"]].astype(float).to_numpy())
    if not days:
        raise ValueError("trades must not be empty")

    pass_days = []
    profits = []
    trades_used = []
    pass_events = 0
    fail_events = 0
    unresolved_events = 0

    for _ in range(simulations):
        equity = INITIAL_CAPITAL
        passed = False
        failed = False
        trade_count = 0

        for calendar_day in range(1, MAX_CALENDAR_DAYS + 1):
            daily_pnl = 0.0
            day_trades = days[int(rng.integers(0, len(days)))]
            for net_r, mae in day_trades[: rules.max_trades_daily]:
                mae_pnl = float(mae) * rules.risk_per_trade * INITIAL_CAPITAL
                if daily_pnl + mae_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                    failed = True
                    break
                if equity + mae_pnl <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                    failed = True
                    break

                pnl = float(net_r) * rules.risk_per_trade * INITIAL_CAPITAL
                equity += pnl
                daily_pnl += pnl
                trade_count += 1
                if daily_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                    failed = True
                    break
                if equity <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                    failed = True
                    break
                if equity >= INITIAL_CAPITAL * (1 + rules.profit_target):
                    passed = True
                    pass_days.append(calendar_day)
                    break
            if passed or failed:
                break

        pass_events += int(passed)
        fail_events += int(failed)
        unresolved_events += int(not passed and not failed)
        profits.append(equity - INITIAL_CAPITAL)
        trades_used.append(trade_count)

    return PortfolioFundingResult(
        scenario=scenario,
        risk=risk,
        simulations=simulations,
        pass_probability=pass_events / simulations,
        fail_probability=fail_events / simulations,
        unresolved_probability=unresolved_events / simulations,
        average_calendar_days_to_pass=float(np.mean(pass_days)) if pass_days else None,
        median_calendar_days_to_pass=float(np.median(pass_days)) if pass_days else None,
        expected_profit=float(np.mean(profits)),
        average_trades_used=float(np.mean(trades_used)),
    )


def add_h4_context(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    h4 = resample_ohlcv(enriched, "4h")
    h4["h4_ema50"] = h4["close"].ewm(span=50, adjust=False).mean()
    h4["h4_slope"] = h4["h4_ema50"].diff()
    h4_context = h4[["datetime", "h4_ema50", "h4_slope"]].rename(columns={"datetime": "h4_time"})
    enriched["h4_time"] = pd.to_datetime(enriched["datetime"]).dt.floor("4h")
    enriched = enriched.merge(h4_context, on="h4_time", how="left")
    return enriched.drop(columns=["h4_time"])


def write_summary(summary: pd.DataFrame, folds: pd.DataFrame, funding: pd.DataFrame) -> None:
    lines = [
        "# Yahoo Index H1 Mean Reversion Study",
        "",
        "Proxy data: Yahoo Finance H1. Costs are simulated in index points, not broker-verified CFD costs.",
        "",
        "Rules:",
        "",
        "- H1 long-only EMA50 mean reversion",
        "- Entry: close <= EMA50 - 2.25 ATR",
        "- Filter: H4 EMA50 slope positive",
        "- SL 2 ATR, TP 2R, max 16 H1 bars",
        "- Walk-forward: 6 months train, 3 months test",
        "",
        "## Asset Summary",
        "",
    ]
    if summary.empty:
        lines.append("No trades generated.")
    else:
        lines.append(summary.sort_values(["scenario", "expectancy"], ascending=[True, False]).to_markdown(index=False))

    lines.extend(["", "## Portfolio FTMO Proxy", ""])
    if funding.empty:
        lines.append("No funding simulation generated.")
    else:
        lines.append(funding.sort_values(["scenario", "risk"]).to_markdown(index=False))

    lines.extend(["", "## Walk-Forward Folds", ""])
    if folds.empty:
        lines.append("No folds generated.")
    else:
        lines.append(folds.to_markdown(index=False))
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
