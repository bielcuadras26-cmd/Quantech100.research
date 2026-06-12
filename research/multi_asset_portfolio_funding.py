"""Portfolio-level prop-firm simulation for the multi-asset reversion study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from research.multi_asset_reversion_comparison import OUTPUT_DIR as SOURCE_DIR


OUTPUT_DIR = Path("research/results/multi_asset_portfolio_funding")
INITIAL_CAPITAL = 100_000.0
MAX_CALENDAR_DAYS = 90
SCENARIOS = ["p95_realistic", "stress"]
EMPTY_DAY = np.empty((0, 2), dtype=float)
PORTFOLIOS = {
    "US100": ["US100"],
    "EURUSD+US100": ["EURUSD", "US100"],
    "ALL": ["EURUSD", "XAUUSD", "US100"],
}


@dataclass(frozen=True)
class PortfolioFundingResult:
    portfolio: str
    scenario: str
    preset: str
    risk: float
    simulations: int
    assets: str
    pass_probability: float
    fail_probability: float
    unresolved_probability: float
    average_calendar_days_to_pass: float | None
    median_calendar_days_to_pass: float | None
    average_calendar_days_to_fail: float | None
    average_trading_days_to_pass: float | None
    median_trading_days_to_pass: float | None
    expected_profit: float
    average_trades_used: float
    average_daily_trades: float
    average_daily_realized_loss_breaches: float
    average_daily_mae_loss_breaches: float


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for scenario in SCENARIOS:
        asset_days = {
            asset: build_asset_calendar(load_asset_trades(asset, scenario))
            for asset in ["EURUSD", "XAUUSD", "US100"]
        }
        for portfolio, assets in PORTFOLIOS.items():
            for risk in [0.0025, 0.005, 0.0075]:
                result = simulate_portfolio_funding(
                    asset_days,
                    portfolio=portfolio,
                    assets=assets,
                    scenario=scenario,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    risk=risk,
                    simulations=5_000,
                )
                all_results.append(result.__dict__)

    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_DIR / "portfolio_funding_simulations.csv", index=False)
    write_summary(results)


def load_asset_trades(asset: str, scenario: str) -> pd.DataFrame:
    path = SOURCE_DIR / f"{asset.lower()}_{scenario}_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run research/multi_asset_reversion_comparison.py first: missing {path}")
    trades = pd.read_csv(path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["trade_date"] = trades["entry_time"].dt.date
    trades["asset"] = asset
    return trades


def build_asset_calendar(trades: pd.DataFrame) -> list[np.ndarray]:
    start = pd.Timestamp(trades["entry_time"].min()).normalize()
    end = pd.Timestamp(trades["entry_time"].max()).normalize()
    grouped = {date: group.copy() for date, group in trades.groupby("trade_date", sort=True)}
    days = []
    for date in pd.date_range(start, end, freq="D"):
        day = grouped.get(date.date())
        if day is None or day.empty:
            days.append(EMPTY_DAY)
            continue
        day = day.sort_values(["entry_time", "exit_time"])
        days.append(day[["net_r", "mae"]].astype(float).to_numpy())
    return days


def simulate_portfolio_funding(
    asset_days: dict[str, list[np.ndarray]],
    *,
    portfolio: str,
    assets: list[str],
    scenario: str,
    rules: PropFirmRules,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> PortfolioFundingResult:
    rng = np.random.default_rng(seed)
    pass_days = []
    fail_days = []
    pass_trading_days = []
    profits = []
    trades_used = []
    daily_trade_counts = []
    realized_daily_breaches = 0
    mae_daily_breaches = 0
    pass_events = 0
    fail_events = 0
    unresolved_events = 0

    for _ in range(simulations):
        equity = INITIAL_CAPITAL
        passed = False
        failed = False
        trading_days = 0
        trade_count = 0

        for calendar_day in range(1, MAX_CALENDAR_DAYS + 1):
            day_trades = sample_portfolio_day(asset_days, assets, rng)
            if len(day_trades) == 0:
                daily_trade_counts.append(0)
                continue

            trading_days += 1
            daily_pnl = 0.0
            taken_today = 0

            for net_r, mae in day_trades[: rules.max_trades_daily]:
                mae_pnl = float(mae) * rules.risk_per_trade * INITIAL_CAPITAL
                if daily_pnl + mae_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                    failed = True
                    mae_daily_breaches += 1
                    fail_days.append(calendar_day)
                    break
                if equity + mae_pnl <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                    failed = True
                    fail_days.append(calendar_day)
                    break

                pnl = float(net_r) * rules.risk_per_trade * INITIAL_CAPITAL
                equity += pnl
                daily_pnl += pnl
                trade_count += 1
                taken_today += 1

                if daily_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                    failed = True
                    realized_daily_breaches += 1
                    fail_days.append(calendar_day)
                    break
                if equity <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                    failed = True
                    fail_days.append(calendar_day)
                    break
                if equity >= INITIAL_CAPITAL * (1 + rules.profit_target):
                    passed = True
                    pass_days.append(calendar_day)
                    pass_trading_days.append(trading_days)
                    break

            daily_trade_counts.append(taken_today)
            if passed or failed:
                break

        pass_events += int(passed)
        fail_events += int(failed)
        unresolved_events += int(not passed and not failed)
        profits.append(equity - INITIAL_CAPITAL)
        trades_used.append(trade_count)

    return PortfolioFundingResult(
        portfolio=portfolio,
        scenario=scenario,
        preset=rules.name,
        risk=risk,
        simulations=simulations,
        assets="+".join(assets),
        pass_probability=pass_events / simulations,
        fail_probability=fail_events / simulations,
        unresolved_probability=unresolved_events / simulations,
        average_calendar_days_to_pass=float(np.mean(pass_days)) if pass_days else None,
        median_calendar_days_to_pass=float(np.median(pass_days)) if pass_days else None,
        average_calendar_days_to_fail=float(np.mean(fail_days)) if fail_days else None,
        average_trading_days_to_pass=float(np.mean(pass_trading_days)) if pass_trading_days else None,
        median_trading_days_to_pass=float(np.median(pass_trading_days)) if pass_trading_days else None,
        expected_profit=float(np.mean(profits)),
        average_trades_used=float(np.mean(trades_used)),
        average_daily_trades=float(np.mean(daily_trade_counts)),
        average_daily_realized_loss_breaches=realized_daily_breaches / simulations,
        average_daily_mae_loss_breaches=mae_daily_breaches / simulations,
    )


def sample_portfolio_day(
    asset_days: dict[str, list[np.ndarray]],
    assets: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_days = []
    for asset in assets:
        calendar = asset_days[asset]
        day = calendar[int(rng.integers(0, len(calendar)))]
        if len(day) > 0:
            sampled_days.append(day)

    if not sampled_days:
        return EMPTY_DAY

    return np.concatenate(sampled_days, axis=0)


def write_summary(results: pd.DataFrame) -> None:
    display_cols = [
        "portfolio",
        "scenario",
        "risk",
        "pass_probability",
        "fail_probability",
        "unresolved_probability",
        "average_calendar_days_to_pass",
        "median_calendar_days_to_pass",
        "average_trading_days_to_pass",
        "expected_profit",
        "average_trades_used",
        "average_daily_trades",
        "average_daily_mae_loss_breaches",
    ]
    lines = [
        "# Multi-Asset Portfolio Funding Simulation",
        "",
        "Rules:",
        "",
        "- FTMO-style 10% target, 5% max daily loss, 10% max total loss",
        "- 90 calendar-day evaluation window",
        "- Max 8 trades per day across the whole portfolio",
        "- Daily loss checks include closed PnL and intratrade MAE stress",
        "- Trades come from walk-forward test trades generated by the multi-asset study",
        "- Calendar sampling includes no-trade days from each asset history",
        "",
        "## Results",
        "",
        results[display_cols].sort_values(["scenario", "risk", "pass_probability"], ascending=[True, True, False]).to_markdown(index=False),
        "",
        "## Best By Scenario And Risk",
        "",
    ]
    best = results.sort_values(
        ["scenario", "risk", "pass_probability", "fail_probability"],
        ascending=[True, True, False, True],
    ).groupby(["scenario", "risk"], as_index=False).head(1)
    lines.append(best[display_cols].to_markdown(index=False))
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
