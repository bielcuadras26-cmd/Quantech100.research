"""Aggressive funding ROI study for the US100 + XAUUSD portfolio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from research.multi_asset_portfolio_funding import (
    INITIAL_CAPITAL,
    MAX_CALENDAR_DAYS,
    build_asset_calendar,
    sample_portfolio_day,
    simulate_portfolio_funding,
)
from research.top_two_portfolio_funding_study import load_trades


OUTPUT_DIR = Path("research/results/aggressive_funding_roi")
SCENARIOS = ["p95_realistic", "stress"]
RISKS = [0.005, 0.0075, 0.01, 0.0125, 0.015]
SIMULATIONS = 3_000
YEAR_SIMULATIONS = 2_000
YEAR_DAYS = 365
CHALLENGE_FEE = 500.0
EXPECTED_PAYOUT_PER_PASS = 9_000.0

TRADE_PATHS = {
    "US100": {
        "p95_realistic": Path("research/results/multi_asset_reversion/us100_p95_realistic_trades.csv"),
        "stress": Path("research/results/multi_asset_reversion/us100_stress_trades.csv"),
    },
    "XAUUSD": {
        "p95_realistic": Path("research/results/multi_asset_reversion/xauusd_p95_realistic_trades.csv"),
        "stress": Path("research/results/multi_asset_reversion/xauusd_stress_trades.csv"),
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    funding_rows = []
    annual_rows = []

    for scenario in SCENARIOS:
        trades = {asset: load_trades(asset, paths[scenario]) for asset, paths in TRADE_PATHS.items()}
        asset_days = {asset: build_asset_calendar(book) for asset, book in trades.items()}
        for risk in RISKS:
            rules = replace(PRESETS["FTMO"], risk_per_trade=risk)
            funding = simulate_portfolio_funding(
                asset_days,
                portfolio="US100+XAUUSD",
                assets=["US100", "XAUUSD"],
                scenario=scenario,
                rules=rules,
                risk=risk,
                simulations=SIMULATIONS,
            )
            funding_rows.append(funding.__dict__)
            annual_rows.append(
                simulate_annual_roi(
                    asset_days,
                    scenario=scenario,
                    rules=rules,
                    risk=risk,
                    simulations=YEAR_SIMULATIONS,
                )
            )

    funding_df = pd.DataFrame(funding_rows)
    annual_df = pd.DataFrame(annual_rows)
    funding_df.to_csv(OUTPUT_DIR / "aggressive_90_day_funding.csv", index=False)
    annual_df.to_csv(OUTPUT_DIR / "aggressive_annual_roi.csv", index=False)
    write_summary(funding_df, annual_df)


def simulate_annual_roi(
    asset_days: dict[str, list[np.ndarray]],
    *,
    scenario: str,
    rules: PropFirmRules,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    passes = []
    fails = []
    expired = []
    started = []
    net_profit = []
    roi = []

    for _ in range(simulations):
        day_of_year = 0
        year_passes = 0
        year_fails = 0
        year_expired = 0
        year_started = 0

        while day_of_year < YEAR_DAYS:
            outcome, days_used = run_single_challenge(asset_days, rules, rng)
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

        gross_payout = year_passes * EXPECTED_PAYOUT_PER_PASS
        fees = year_started * CHALLENGE_FEE
        profit = gross_payout - fees
        passes.append(year_passes)
        fails.append(year_fails)
        expired.append(year_expired)
        started.append(year_started)
        net_profit.append(profit)
        roi.append(profit / fees if fees > 0 else 0.0)

    return {
        "portfolio": "US100+XAUUSD",
        "scenario": scenario,
        "risk": risk,
        "simulations": simulations,
        "average_challenges_started": float(np.mean(started)),
        "average_passed": float(np.mean(passes)),
        "median_passed": float(np.median(passes)),
        "average_failed": float(np.mean(fails)),
        "median_failed": float(np.median(fails)),
        "average_expired_without_pass": float(np.mean(expired)),
        "probability_at_least_1_pass": float(np.mean(np.array(passes) >= 1)),
        "probability_at_least_2_passes": float(np.mean(np.array(passes) >= 2)),
        "probability_at_least_3_passes": float(np.mean(np.array(passes) >= 3)),
        "expected_gross_payout": float(np.mean(passes) * EXPECTED_PAYOUT_PER_PASS),
        "expected_fees": float(np.mean(started) * CHALLENGE_FEE),
        "expected_net_profit": float(np.mean(net_profit)),
        "expected_roi_on_fees": float(np.mean(roi)),
        "probability_losing_year": float(np.mean(np.array(net_profit) < 0)),
    }


def run_single_challenge(
    asset_days: dict[str, list[np.ndarray]],
    rules: PropFirmRules,
    rng: np.random.Generator,
) -> tuple[str, int]:
    equity = INITIAL_CAPITAL
    for calendar_day in range(1, MAX_CALENDAR_DAYS + 1):
        day_trades = sample_portfolio_day(asset_days, ["US100", "XAUUSD"], rng)
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


def write_summary(funding: pd.DataFrame, annual: pd.DataFrame) -> None:
    funding_cols = [
        "scenario",
        "risk",
        "pass_probability",
        "fail_probability",
        "unresolved_probability",
        "average_calendar_days_to_pass",
        "expected_profit",
        "average_trades_used",
        "average_daily_trades",
    ]
    annual_cols = [
        "scenario",
        "risk",
        "average_challenges_started",
        "average_passed",
        "median_passed",
        "average_failed",
        "average_expired_without_pass",
        "expected_gross_payout",
        "expected_fees",
        "expected_net_profit",
        "expected_roi_on_fees",
        "probability_losing_year",
    ]
    lines = [
        "# Aggressive Funding ROI Study",
        "",
        "- Portfolio: US100 + XAUUSD",
        f"- Challenge fee assumption: ${CHALLENGE_FEE:,.0f}",
        f"- Expected payout per approved funded account: ${EXPECTED_PAYOUT_PER_PASS:,.0f}",
        "- A challenge that reaches 90 days without pass/fail is treated as expired and a new challenge starts.",
        "",
        "## 90-Day Challenge Results",
        "",
        funding[funding_cols].sort_values(["scenario", "risk"]).to_markdown(index=False),
        "",
        "## Annual ROI Results",
        "",
        annual[annual_cols].sort_values(["scenario", "risk"]).to_markdown(index=False),
        "",
        "## Best By Expected Net Profit",
        "",
        annual.sort_values("expected_net_profit", ascending=False)[annual_cols].head(6).to_markdown(index=False),
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
