"""Compare US100 with the strongest secondary candidates in funding simulations."""

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


OUTPUT_DIR = Path("research/results/top_two_portfolio_funding")
SIMULATIONS = 2_000
YEAR_SIMULATIONS = 1_000
YEAR_DAYS = 365
SCENARIOS = ["p95_realistic", "stress"]
RISKS = [0.005, 0.0075]

CANDIDATES = {
    "US100": {
        "p95_realistic": Path("research/results/multi_asset_reversion/us100_p95_realistic_trades.csv"),
        "stress": Path("research/results/multi_asset_reversion/us100_stress_trades.csv"),
    },
    "XAUUSD": {
        "p95_realistic": Path("research/results/multi_asset_reversion/xauusd_p95_realistic_trades.csv"),
        "stress": Path("research/results/multi_asset_reversion/xauusd_stress_trades.csv"),
    },
    "DE40": {
        "p95_realistic": Path("research/results/mt5_index_m15_reversion/de40_p95_realistic_base_trades.csv"),
        "stress": Path("research/results/mt5_index_m15_reversion/de40_stress_base_trades.csv"),
    },
    "USDJPY_SHORT": {
        "p95_realistic": Path("research/results/usdjpy_m15_reversion/usdjpy_p95_realistic_candidate_short_trades.csv"),
        "stress": Path("research/results/usdjpy_m15_reversion/usdjpy_stress_candidate_short_trades.csv"),
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    funding_rows = []
    annual_rows = []

    for scenario in SCENARIOS:
        loaded = {asset: load_trades(asset, paths[scenario]) for asset, paths in CANDIDATES.items()}
        for secondary in ["XAUUSD", "DE40", "USDJPY_SHORT"]:
            portfolio_name = f"US100+{secondary}"
            assets = ["US100", secondary]
            asset_days = {asset: build_asset_calendar(loaded[asset]) for asset in assets}
            for risk in RISKS:
                result = simulate_portfolio_funding(
                    asset_days,
                    portfolio=portfolio_name,
                    assets=assets,
                    scenario=scenario,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    risk=risk,
                    simulations=SIMULATIONS,
                )
                funding_rows.append(result.__dict__)
                annual_rows.append(
                    simulate_annual_churn(
                        asset_days,
                        portfolio=portfolio_name,
                        assets=assets,
                        scenario=scenario,
                        rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                        risk=risk,
                        simulations=YEAR_SIMULATIONS,
                    )
                )

    funding = pd.DataFrame(funding_rows)
    annual = pd.DataFrame(annual_rows)
    funding.to_csv(OUTPUT_DIR / "pair_90_day_funding.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "pair_annual_churn.csv", index=False)
    write_summary(funding, annual)


def load_trades(asset: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trades for {asset}: {path}")
    trades = pd.read_csv(path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["trade_date"] = trades["entry_time"].dt.date
    trades["asset"] = asset
    return trades


def simulate_annual_churn(
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


def write_summary(funding: pd.DataFrame, annual: pd.DataFrame) -> None:
    funding_cols = [
        "portfolio",
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
        "portfolio",
        "scenario",
        "risk",
        "average_challenges_started",
        "average_passed",
        "median_passed",
        "average_failed",
        "median_failed",
        "average_expired_without_pass",
        "probability_at_least_1_pass",
        "probability_at_least_2_passes",
        "probability_at_least_3_passes",
    ]
    best = annual.sort_values(
        ["scenario", "risk", "average_passed", "average_failed"],
        ascending=[True, True, False, True],
    ).groupby(["scenario", "risk"], as_index=False).head(1)
    lines = [
        "# Top Two Portfolio Funding Study",
        "",
        "US100 is compared with XAUUSD, DE40, and the USDJPY short-only candidate.",
        "",
        "## 90-Day Funding Simulation",
        "",
        funding[funding_cols].sort_values(["scenario", "risk", "pass_probability"], ascending=[True, True, False]).to_markdown(index=False),
        "",
        "## Annual Sequential Churn",
        "",
        annual[annual_cols].sort_values(["scenario", "risk", "average_passed"], ascending=[True, True, False]).to_markdown(index=False),
        "",
        "## Best Pair By Scenario And Risk",
        "",
        best[annual_cols].to_markdown(index=False),
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
