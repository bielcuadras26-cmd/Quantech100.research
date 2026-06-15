"""Aggressive funding ROI with a defensive daily stop before the prop limit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from research.aggressive_funding_roi_study import (
    CHALLENGE_FEE,
    EXPECTED_PAYOUT_PER_PASS,
    RISKS,
    SCENARIOS,
    TRADE_PATHS,
    YEAR_DAYS,
)
from research.multi_asset_portfolio_funding import (
    INITIAL_CAPITAL,
    MAX_CALENDAR_DAYS,
    build_asset_calendar,
    sample_portfolio_day,
)
from research.top_two_portfolio_funding_study import load_trades


OUTPUT_DIR = Path("research/results/defensive_daily_stop_roi")
SIMULATIONS = 3_000
YEAR_SIMULATIONS = 2_000
DAILY_STOP = 0.03


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    challenge_rows = []
    annual_rows = []
    for scenario in SCENARIOS:
        trades = {asset: load_trades(asset, paths[scenario]) for asset, paths in TRADE_PATHS.items()}
        asset_days = {asset: build_asset_calendar(book) for asset, book in trades.items()}
        for risk in RISKS:
            rules = replace(PRESETS["FTMO"], risk_per_trade=risk)
            challenge_rows.append(
                simulate_challenges(asset_days, scenario=scenario, rules=rules, risk=risk, simulations=SIMULATIONS)
            )
            annual_rows.append(
                simulate_annual_roi(asset_days, scenario=scenario, rules=rules, risk=risk, simulations=YEAR_SIMULATIONS)
            )

    challenges = pd.DataFrame(challenge_rows)
    annual = pd.DataFrame(annual_rows)
    challenges.to_csv(OUTPUT_DIR / "defensive_90_day_funding.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "defensive_annual_roi.csv", index=False)
    write_summary(challenges, annual)


def simulate_challenges(
    asset_days: dict[str, list[np.ndarray]],
    *,
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
    pass_days = []
    stopped_days = []
    profits = []

    for _ in range(simulations):
        outcome, days_used, stopped, equity = run_single_challenge(asset_days, rules, rng)
        passed.append(int(outcome == "passed"))
        failed.append(int(outcome == "failed"))
        expired.append(int(outcome == "expired"))
        if outcome == "passed":
            pass_days.append(days_used)
        stopped_days.append(stopped)
        profits.append(equity - INITIAL_CAPITAL)

    return {
        "portfolio": "US100+XAUUSD",
        "scenario": scenario,
        "risk": risk,
        "daily_stop": DAILY_STOP,
        "simulations": simulations,
        "pass_probability": float(np.mean(passed)),
        "fail_probability": float(np.mean(failed)),
        "unresolved_probability": float(np.mean(expired)),
        "average_calendar_days_to_pass": float(np.mean(pass_days)) if pass_days else None,
        "average_stopped_days": float(np.mean(stopped_days)),
        "expected_profit": float(np.mean(profits)),
    }


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
    stopped_days = []
    net_profit = []
    roi = []

    for _ in range(simulations):
        day_of_year = 0
        year_passes = 0
        year_fails = 0
        year_expired = 0
        year_started = 0
        year_stopped_days = 0

        while day_of_year < YEAR_DAYS:
            outcome, days_used, stopped, _equity = run_single_challenge(asset_days, rules, rng)
            if day_of_year + days_used > YEAR_DAYS:
                break
            day_of_year += days_used
            year_started += 1
            year_stopped_days += stopped
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
        stopped_days.append(year_stopped_days)
        net_profit.append(profit)
        roi.append(profit / fees if fees > 0 else 0.0)

    return {
        "portfolio": "US100+XAUUSD",
        "scenario": scenario,
        "risk": risk,
        "daily_stop": DAILY_STOP,
        "simulations": simulations,
        "average_challenges_started": float(np.mean(started)),
        "average_passed": float(np.mean(passes)),
        "median_passed": float(np.median(passes)),
        "average_failed": float(np.mean(fails)),
        "median_failed": float(np.median(fails)),
        "average_expired_without_pass": float(np.mean(expired)),
        "average_stopped_days": float(np.mean(stopped_days)),
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
) -> tuple[str, int, int, float]:
    equity = INITIAL_CAPITAL
    stopped_days = 0
    for calendar_day in range(1, MAX_CALENDAR_DAYS + 1):
        day_trades = sample_portfolio_day(asset_days, ["US100", "XAUUSD"], rng)
        daily_pnl = 0.0
        stopped_today = False
        for net_r, mae in day_trades[: rules.max_trades_daily]:
            mae_pnl = float(mae) * rules.risk_per_trade * INITIAL_CAPITAL
            if daily_pnl + mae_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                return "failed", calendar_day, stopped_days, equity
            if equity + mae_pnl <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                return "failed", calendar_day, stopped_days, equity
            if daily_pnl + mae_pnl <= -DAILY_STOP * INITIAL_CAPITAL:
                stopped_today = True
                break

            pnl = float(net_r) * rules.risk_per_trade * INITIAL_CAPITAL
            equity += pnl
            daily_pnl += pnl

            if daily_pnl <= -rules.max_daily_loss * INITIAL_CAPITAL:
                return "failed", calendar_day, stopped_days, equity
            if equity <= INITIAL_CAPITAL * (1 - rules.max_total_loss):
                return "failed", calendar_day, stopped_days, equity
            if equity >= INITIAL_CAPITAL * (1 + rules.profit_target):
                return "passed", calendar_day, stopped_days + int(stopped_today), equity
            if daily_pnl <= -DAILY_STOP * INITIAL_CAPITAL:
                stopped_today = True
                break

        stopped_days += int(stopped_today)

    return "expired", MAX_CALENDAR_DAYS, stopped_days, equity


def write_summary(challenges: pd.DataFrame, annual: pd.DataFrame) -> None:
    challenge_cols = [
        "scenario",
        "risk",
        "daily_stop",
        "pass_probability",
        "fail_probability",
        "unresolved_probability",
        "average_calendar_days_to_pass",
        "average_stopped_days",
        "expected_profit",
    ]
    annual_cols = [
        "scenario",
        "risk",
        "daily_stop",
        "average_challenges_started",
        "average_passed",
        "median_passed",
        "average_failed",
        "average_expired_without_pass",
        "average_stopped_days",
        "expected_net_profit",
        "expected_roi_on_fees",
        "probability_losing_year",
    ]
    lines = [
        "# Defensive Daily Stop ROI Study",
        "",
        "- Portfolio: US100 + XAUUSD",
        "- Official max daily loss: 5%",
        f"- Operational daily stop: {DAILY_STOP:.0%}",
        f"- Challenge fee assumption: ${CHALLENGE_FEE:,.0f}",
        f"- Expected payout per approved funded account: ${EXPECTED_PAYOUT_PER_PASS:,.0f}",
        "",
        "## 90-Day Challenge Results",
        "",
        challenges[challenge_cols].sort_values(["scenario", "risk"]).to_markdown(index=False),
        "",
        "## Annual ROI Results",
        "",
        annual[annual_cols].sort_values(["scenario", "risk"]).to_markdown(index=False),
        "",
        "## Best By Stress Net Profit",
        "",
        annual[annual["scenario"] == "stress"].sort_values("expected_net_profit", ascending=False)[annual_cols].to_markdown(index=False),
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
