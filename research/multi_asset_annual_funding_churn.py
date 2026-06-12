"""One-year sequential prop-firm churn simulation for the combined portfolio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from research.multi_asset_portfolio_funding import (
    INITIAL_CAPITAL,
    MAX_CALENDAR_DAYS,
    SCENARIOS,
    build_asset_calendar,
    load_asset_trades,
    sample_portfolio_day,
)


OUTPUT_DIR = Path("research/results/multi_asset_annual_funding_churn")
YEAR_DAYS = 365
SIMULATIONS = 500
PORTFOLIOS = {"ALL": ["EURUSD", "XAUUSD", "US100"]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        asset_days = {
            asset: build_asset_calendar(load_asset_trades(asset, scenario))
            for asset in ["EURUSD", "XAUUSD", "US100"]
        }
        for portfolio, assets in PORTFOLIOS.items():
            for risk in [0.0025, 0.005, 0.0075]:
                rows.append(
                    simulate_year(
                        asset_days,
                        portfolio=portfolio,
                        assets=assets,
                        scenario=scenario,
                        rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                        risk=risk,
                        simulations=SIMULATIONS,
                    )
                )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "annual_funding_churn.csv", index=False)
    write_summary(results)


def simulate_year(
    asset_days: dict[str, list[np.ndarray]],
    *,
    portfolio: str,
    assets: list[str],
    scenario: str,
    rules: PropFirmRules,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    passed = []
    failed = []
    expired = []
    started = []
    active_days = []
    net_fee_units = []

    for _ in range(simulations):
        day_of_year = 0
        year_passes = 0
        year_fails = 0
        year_expired = 0
        year_started = 0

        while day_of_year < YEAR_DAYS:
            outcome, days_used = run_single_challenge(asset_days, assets, rules, rng)
            if day_of_year + days_used > YEAR_DAYS:
                break
            day_of_year += days_used
            year_started += 1

            if outcome == "passed":
                year_passes += 1
            elif outcome == "failed":
                year_fails += 1
            else:
                year_expired += 1

        passed.append(year_passes)
        failed.append(year_fails)
        expired.append(year_expired)
        started.append(year_started)
        active_days.append(day_of_year)
        net_fee_units.append(year_passes - year_fails - year_expired)

    return {
        "portfolio": portfolio,
        "scenario": scenario,
        "risk": risk,
        "simulations": simulations,
        "average_challenges_started": float(np.mean(started)),
        "average_passed": float(np.mean(passed)),
        "median_passed": float(np.median(passed)),
        "probability_at_least_1_pass": float(np.mean(np.array(passed) >= 1)),
        "probability_at_least_2_passes": float(np.mean(np.array(passed) >= 2)),
        "probability_at_least_3_passes": float(np.mean(np.array(passed) >= 3)),
        "average_failed": float(np.mean(failed)),
        "median_failed": float(np.median(failed)),
        "average_expired_without_pass": float(np.mean(expired)),
        "median_expired_without_pass": float(np.median(expired)),
        "average_active_days": float(np.mean(active_days)),
        "average_pass_minus_fail_expired": float(np.mean(net_fee_units)),
    }


def run_single_challenge(
    asset_days: dict[str, list[np.ndarray]],
    assets: list[str],
    rules: PropFirmRules,
    rng: np.random.Generator,
) -> tuple[str, int]:
    equity = INITIAL_CAPITAL

    for calendar_day in range(1, MAX_CALENDAR_DAYS + 1):
        day_trades = sample_portfolio_day(asset_days, assets, rng)
        daily_pnl = 0.0

        for net_r, mae in day_trades[: rules.max_trades_daily]:
            mae_pnl = float(mae) * rules.risk_per_trade * INITIAL_CAPITAL
            if daily_pnl + mae_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                return "failed", calendar_day
            if equity + mae_pnl <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                return "failed", calendar_day

            pnl = float(net_r) * rules.risk_per_trade * INITIAL_CAPITAL
            equity += pnl
            daily_pnl += pnl

            if daily_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                return "failed", calendar_day
            if equity <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                return "failed", calendar_day
            if equity >= INITIAL_CAPITAL * (1 + rules.profit_target):
                return "passed", calendar_day

    return "expired", MAX_CALENDAR_DAYS


def write_summary(results: pd.DataFrame) -> None:
    display_cols = [
        "portfolio",
        "scenario",
        "risk",
        "average_challenges_started",
        "average_passed",
        "median_passed",
        "probability_at_least_1_pass",
        "probability_at_least_2_passes",
        "probability_at_least_3_passes",
        "average_failed",
        "average_expired_without_pass",
        "average_pass_minus_fail_expired",
    ]
    lines = [
        "# Annual Sequential Funding Churn",
        "",
        "Assumption: each challenge can last up to 90 calendar days. After pass, fail, or 90-day expiry without pass, a new challenge starts immediately if there is enough time left in the 365-day year.",
        "",
        "## Results",
        "",
        results[display_cols].sort_values(["scenario", "risk", "average_passed"], ascending=[True, True, False]).to_markdown(index=False),
        "",
        "## Best By Scenario And Risk",
        "",
    ]
    best = results.sort_values(
        ["scenario", "risk", "average_passed", "average_failed"],
        ascending=[True, True, False, True],
    ).groupby(["scenario", "risk"], as_index=False).head(1)
    lines.append(best[display_cols].to_markdown(index=False))
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
