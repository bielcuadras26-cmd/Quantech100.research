"""Walk-forward and funding pass-duration study for the EURUSD M15 candidate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.cost_model import CostConfig
from backend.core.data_loader import load_mt5_export
from backend.core.metrics import calculate_metrics
from backend.core.prop_firm_simulator import PRESETS, PropFirmRules
from research.eurusd_m15_reversion_refinement import add_context, apply_filters, resample_ohlcv
from research.eurusd_m5_mean_reversion_study import SOURCE, evaluate_variant


OUTPUT_DIR = Path("research/results/eurusd_m15_walk_forward")

CANDIDATE_PARAMS = {
    "ema_window": 50,
    "atr_window": 14,
    "distance_atr": 2.25,
    "exit_distance_atr": 0.25,
    "stop_atr": 2.0,
    "take_profit_r": 2.0,
    "max_bars": 32,
    "side": "long_only",
    "cost_case": "conservative",
}


@dataclass(frozen=True)
class FundingDurationResult:
    preset: str
    risk: float
    simulations: int
    pass_probability: float
    fail_probability: float
    unresolved_probability: float
    average_days_to_pass: float | None
    median_days_to_pass: float | None
    average_days_to_fail: float | None
    expected_profit: float


@dataclass(frozen=True)
class CalendarFundingDurationResult:
    preset: str
    risk: float
    simulations: int
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = load_mt5_export(SOURCE)
    m15 = add_context(resample_ohlcv(source, "15min"))
    filtered = apply_filters(m15, "all", "h1_slope_up", "none")
    filtered.to_csv(OUTPUT_DIR / "candidate_filtered_data.csv", index=False)

    costs = CostConfig(spread=0.00002, commission_per_unit=0.000035, slippage=0.00001)
    folds = build_quarterly_folds(filtered, train_months=6, test_months=3)
    rows = []
    fold_trade_books = []
    for fold in folds:
        train = filtered[(filtered["datetime"] >= fold["train_start"]) & (filtered["datetime"] < fold["train_end"])]
        test = filtered[(filtered["datetime"] >= fold["test_start"]) & (filtered["datetime"] < fold["test_end"])]
        if len(train) < 300 or len(test) < 100:
            continue
        row, trades = evaluate_variant(
            train.reset_index(drop=True),
            test.reset_index(drop=True),
            CANDIDATE_PARAMS,
            costs,
        )
        row.update(fold)
        rows.append(row)
        if trades is not None and not trades.empty:
            trades = trades.copy()
            trades["fold"] = fold["fold"]
            fold_trade_books.append(trades)

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(OUTPUT_DIR / "walk_forward_folds.csv", index=False)
    all_test_trades = pd.concat(fold_trade_books, ignore_index=True) if fold_trade_books else pd.DataFrame()
    all_test_trades.to_csv(OUTPUT_DIR / "walk_forward_test_trades.csv", index=False)

    funding_rows = []
    calendar_funding_rows = []
    if not all_test_trades.empty:
        for preset in ["FTMO", "FundingPips", "Alpha Capital", "FundingNext"]:
            for risk in [0.0025, 0.005, 0.0075, 0.01]:
                result = simulate_funding_duration(
                    all_test_trades,
                    rules=replace(PRESETS[preset], risk_per_trade=risk),
                    preset=preset,
                    risk=risk,
                    simulations=5_000,
                )
                funding_rows.append(result.__dict__)
                calendar_result = simulate_calendar_funding_duration(
                    all_test_trades,
                    rules=replace(PRESETS[preset], risk_per_trade=risk),
                    preset=preset,
                    risk=risk,
                    simulations=5_000,
                )
                calendar_funding_rows.append(calendar_result.__dict__)

    funding_results = pd.DataFrame(funding_rows)
    funding_results.to_csv(OUTPUT_DIR / "funding_duration_simulations.csv", index=False)
    calendar_funding_results = pd.DataFrame(calendar_funding_rows)
    calendar_funding_results.to_csv(OUTPUT_DIR / "calendar_funding_duration_simulations.csv", index=False)
    write_summary(fold_results, all_test_trades, funding_results, calendar_funding_results)


def build_quarterly_folds(
    data: pd.DataFrame,
    *,
    train_months: int,
    test_months: int,
) -> list[dict[str, pd.Timestamp | str]]:
    start = pd.Timestamp(data["datetime"].min()).to_period("M").to_timestamp()
    end = pd.Timestamp(data["datetime"].max())
    folds = []
    fold = 1
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append(
            {
                "fold": f"fold_{fold}",
                "train_start": train_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            }
        )
        train_start += pd.DateOffset(months=test_months)
        fold += 1
    return folds


def simulate_funding_duration(
    trades: pd.DataFrame,
    *,
    rules: PropFirmRules,
    preset: str,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> FundingDurationResult:
    rng = np.random.default_rng(seed)
    net_r = trades["net_r"].astype(float).to_numpy()
    if len(net_r) == 0:
        raise ValueError("trades must not be empty")

    pass_days = []
    fail_days = []
    profits = []
    pass_events = 0
    fail_events = 0
    unresolved_events = 0
    max_days = 60

    for _ in range(simulations):
        equity = 100_000.0
        day = 1
        daily_pnl = 0.0
        daily_trades = 0
        passed = False
        failed = False
        outcomes = rng.choice(net_r, size=max_days * rules.max_trades_daily, replace=True)

        for outcome in outcomes:
            pnl = float(outcome) * rules.risk_per_trade * 100_000.0
            equity += pnl
            daily_pnl += pnl
            daily_trades += 1

            if daily_pnl <= -rules.max_daily_loss * 100_000.0:
                failed = True
                fail_days.append(day)
                break
            if equity <= 100_000.0 * (1 - rules.max_total_loss):
                failed = True
                fail_days.append(day)
                break
            if equity >= 100_000.0 * (1 + rules.profit_target):
                passed = True
                pass_days.append(day)
                break

            if daily_trades >= rules.max_trades_daily:
                day += 1
                daily_trades = 0
                daily_pnl = 0.0
                if day > max_days:
                    break

        pass_events += int(passed)
        fail_events += int(failed)
        unresolved_events += int(not passed and not failed)
        profits.append(equity - 100_000.0)

    return FundingDurationResult(
        preset=preset,
        risk=risk,
        simulations=simulations,
        pass_probability=pass_events / simulations,
        fail_probability=fail_events / simulations,
        unresolved_probability=unresolved_events / simulations,
        average_days_to_pass=float(np.mean(pass_days)) if pass_days else None,
        median_days_to_pass=float(np.median(pass_days)) if pass_days else None,
        average_days_to_fail=float(np.mean(fail_days)) if fail_days else None,
        expected_profit=float(np.mean(profits)),
    )


def simulate_calendar_funding_duration(
    trades: pd.DataFrame,
    *,
    rules: PropFirmRules,
    preset: str,
    risk: float,
    simulations: int,
    seed: int = 42,
) -> CalendarFundingDurationResult:
    rng = np.random.default_rng(seed)
    prepared = trades.copy()
    prepared["exit_time"] = pd.to_datetime(prepared["exit_time"])
    prepared["trade_date"] = prepared["exit_time"].dt.date
    daily_groups = [
        group["net_r"].astype(float).to_numpy()
        for _, group in prepared.groupby("trade_date", sort=True)
        if not group.empty
    ]
    if not daily_groups:
        raise ValueError("trades must contain at least one dated trade")

    max_calendar_days = 90
    pass_calendar_days = []
    fail_calendar_days = []
    pass_trading_days = []
    trades_used = []
    profits = []
    pass_events = 0
    fail_events = 0
    unresolved_events = 0

    for _ in range(simulations):
        equity = 100_000.0
        passed = False
        failed = False
        trading_days = 0
        trade_count = 0

        for calendar_day in range(1, max_calendar_days + 1):
            day_trades = daily_groups[int(rng.integers(0, len(daily_groups)))]
            if len(day_trades) == 0:
                continue
            trading_days += 1
            daily_pnl = 0.0

            for outcome in day_trades[: rules.max_trades_daily]:
                pnl = float(outcome) * rules.risk_per_trade * 100_000.0
                equity += pnl
                daily_pnl += pnl
                trade_count += 1

                if daily_pnl <= -rules.max_daily_loss * 100_000.0:
                    failed = True
                    fail_calendar_days.append(calendar_day)
                    break
                if equity <= 100_000.0 * (1 - rules.max_total_loss):
                    failed = True
                    fail_calendar_days.append(calendar_day)
                    break
                if equity >= 100_000.0 * (1 + rules.profit_target):
                    passed = True
                    pass_calendar_days.append(calendar_day)
                    pass_trading_days.append(trading_days)
                    break
            if passed or failed:
                break

        pass_events += int(passed)
        fail_events += int(failed)
        unresolved_events += int(not passed and not failed)
        profits.append(equity - 100_000.0)
        trades_used.append(trade_count)

    return CalendarFundingDurationResult(
        preset=preset,
        risk=risk,
        simulations=simulations,
        pass_probability=pass_events / simulations,
        fail_probability=fail_events / simulations,
        unresolved_probability=unresolved_events / simulations,
        average_calendar_days_to_pass=float(np.mean(pass_calendar_days)) if pass_calendar_days else None,
        median_calendar_days_to_pass=float(np.median(pass_calendar_days)) if pass_calendar_days else None,
        average_calendar_days_to_fail=float(np.mean(fail_calendar_days)) if fail_calendar_days else None,
        average_trading_days_to_pass=float(np.mean(pass_trading_days)) if pass_trading_days else None,
        median_trading_days_to_pass=float(np.median(pass_trading_days)) if pass_trading_days else None,
        expected_profit=float(np.mean(profits)),
        average_trades_used=float(np.mean(trades_used)),
    )


def write_summary(
    folds: pd.DataFrame,
    trades: pd.DataFrame,
    funding: pd.DataFrame,
    calendar_funding: pd.DataFrame,
) -> None:
    lines = [
        "# EURUSD M15 Walk-Forward Funding Study",
        "",
        "## Candidate",
        "",
        "- M15 long-only EMA50 mean reversion",
        "- Entry: close <= EMA50 - 2.25 ATR",
        "- Filter: H1 EMA50 slope positive",
        "- SL: 2 ATR",
        "- TP: 2R",
        "- Max duration: 32 M15 bars",
        "- Costs: conservative",
        "",
        "## Walk-Forward Folds",
        "",
    ]
    if folds.empty:
        lines.append("No folds generated.")
    else:
        cols = [
            "fold",
            "test_start",
            "test_end",
            "test_trades",
            "test_win_rate",
            "test_profit_factor",
            "test_net_expectancy",
            "test_total_return",
            "test_max_drawdown",
        ]
        lines.append(folds[cols].to_markdown(index=False))
        lines.extend(
            [
                "",
                "## Aggregate Walk-Forward",
                "",
                f"- Folds: {len(folds)}",
                f"- Positive folds: {(folds['test_net_expectancy'] > 0).sum()}",
                f"- Total test trades: {int(folds['test_trades'].sum())}",
                f"- Average test expectancy: {folds['test_net_expectancy'].mean():.4f} R",
                f"- Worst fold expectancy: {folds['test_net_expectancy'].min():.4f} R",
            ]
        )

    lines.extend(["", "## Funding Pass Duration", ""])
    if funding.empty:
        lines.append("No funding simulations generated.")
    else:
        display = funding.sort_values(["preset", "risk"])
        lines.append(display.to_markdown(index=False))

    lines.extend(["", "## Calendar-Real Funding Pass Duration", ""])
    if calendar_funding.empty:
        lines.append("No calendar-real funding simulations generated.")
    else:
        display = calendar_funding.sort_values(["preset", "risk"])
        lines.append(display.to_markdown(index=False))

    if not trades.empty:
        metrics = calculate_metrics(
            trades,
            pd.DataFrame({"datetime": trades["exit_time"], "equity": 100_000 + trades["net_result"].cumsum()}),
            initial_capital=100_000,
        )
        lines.extend(
            [
                "",
                "## Combined Test Trades",
                "",
                f"- Trades: {len(trades)}",
                f"- Net expectancy: {metrics.net_expectancy:.4f} R",
                f"- Profit factor: {metrics.net_profit_factor:.4f}",
                f"- Win rate: {metrics.win_rate:.2%}",
                f"- Max drawdown: {metrics.max_drawdown:.2%}",
            ]
        )

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
