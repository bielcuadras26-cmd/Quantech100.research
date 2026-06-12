"""Focused EURUSD M15 mean-reversion refinement study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from backend.core.cost_model import CostConfig
from backend.core.data_loader import load_mt5_export
from backend.core.indicators import atr, ema, price_to_average_distance_atr
from backend.core.montecarlo import run_monte_carlo
from backend.core.prop_firm_simulator import PRESETS, simulate_prop_firm
from research.eurusd_m5_mean_reversion_study import SOURCE, evaluate_variant


OUTPUT_DIR = Path("research/results/eurusd_m15_refinement")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    m5 = load_mt5_export(SOURCE)
    m15 = resample_ohlcv(m5, "15min")
    m15 = add_context(m15)
    split = int(len(m15) * 0.7)
    train = m15.iloc[:split].reset_index(drop=True)
    test = m15.iloc[split:].reset_index(drop=True)

    costs = CostConfig(spread=0.00002, commission_per_unit=0.000035, slippage=0.00001)
    rows = []
    trade_books: dict[str, pd.DataFrame] = {}

    for session_filter in ["all", "london_ny"]:
        for trend_filter in ["none", "above_h1_ema", "h1_slope_up"]:
            for volatility_filter in ["none", "atr_above_median"]:
                filtered_train = apply_filters(train, session_filter, trend_filter, volatility_filter)
                filtered_test = apply_filters(test, session_filter, trend_filter, volatility_filter)
                if len(filtered_train) < 500 or len(filtered_test) < 200:
                    continue
                for distance_atr in [2.25, 2.5, 2.75]:
                    for stop_atr in [2.0, 2.5, 3.0]:
                        for take_profit_r in [1.0, 1.5, 2.0]:
                            for max_bars in [24, 32, 48]:
                                params = {
                                    "ema_window": 50,
                                    "atr_window": 14,
                                    "distance_atr": distance_atr,
                                    "exit_distance_atr": 0.25,
                                    "stop_atr": stop_atr,
                                    "take_profit_r": take_profit_r,
                                    "max_bars": max_bars,
                                    "side": "long_only",
                                    "cost_case": "conservative",
                                }
                                row, trades = evaluate_variant(filtered_train, filtered_test, params, costs)
                                row["session_filter"] = session_filter
                                row["trend_filter"] = trend_filter
                                row["volatility_filter"] = volatility_filter
                                row["filtered_train_rows"] = len(filtered_train)
                                row["filtered_test_rows"] = len(filtered_test)
                                rows.append(row)
                                trade_books[str(row["variant_id"]) + filter_key(row)] = trades

    results = pd.DataFrame(rows).sort_values(
        ["test_net_expectancy", "test_profit_factor", "test_trades"],
        ascending=[False, False, False],
    )
    results.to_csv(OUTPUT_DIR / "refinement_results.csv", index=False)

    robust = results[
        (results["train_trades"] >= 40)
        & (results["test_trades"] >= 20)
        & (results["train_net_expectancy"] > 0)
        & (results["test_net_expectancy"] > 0)
        & (results["test_profit_factor"] >= 1.15)
        & (results["test_max_drawdown"] > -0.08)
    ].copy()
    robust.to_csv(OUTPUT_DIR / "robust_refinement_candidates.csv", index=False)

    simulations = []
    for _, row in robust.head(12).iterrows():
        key = str(row["variant_id"]) + filter_key(row)
        trades = trade_books.get(key)
        if trades is None or trades.empty:
            continue
        for risk in [0.0025, 0.005, 0.0075, 0.01]:
            mc = run_monte_carlo(
                trades["net_r"],
                simulations=3_000,
                initial_capital=100_000,
                risk_per_trade=risk,
                target_return=0.10,
                max_drawdown_limit=0.10,
            )
            prop = simulate_prop_firm(
                trades["net_r"],
                rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                initial_capital=100_000,
                simulations=3_000,
            )
            simulations.append(
                {
                    "variant_id": row["variant_id"],
                    "session_filter": row["session_filter"],
                    "trend_filter": row["trend_filter"],
                    "volatility_filter": row["volatility_filter"],
                    "test_trades": row["test_trades"],
                    "test_expectancy": row["test_net_expectancy"],
                    "test_profit_factor": row["test_profit_factor"],
                    "test_drawdown": row["test_max_drawdown"],
                    "risk": risk,
                    "mc_p50": mc.percentile_50,
                    "mc_worst_drawdown": mc.worst_drawdown,
                    "mc_limit_breach_probability": mc.limit_breach_probability,
                    "ftmo_pass_probability": prop.pass_probability,
                    "ftmo_fail_probability": prop.fail_probability,
                    "ftmo_expected_profit": prop.expected_profit,
                    "ftmo_withdrawal_probability": prop.withdrawal_probability,
                }
            )

    simulation_results = pd.DataFrame(simulations)
    simulation_results.to_csv(OUTPUT_DIR / "refinement_ftmo_simulations.csv", index=False)
    write_summary(m15, train, test, results, robust, simulation_results)


def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    frame = data.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.set_index("datetime")
    return (
        frame.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )


def add_context(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    enriched["datetime"] = pd.to_datetime(enriched["datetime"])
    enriched["hour"] = enriched["datetime"].dt.hour
    enriched["ema50"] = ema(enriched["close"], 50)
    enriched["atr14"] = atr(enriched["high"], enriched["low"], enriched["close"], 14)
    enriched["distance_atr"] = price_to_average_distance_atr(
        enriched["close"],
        enriched["ema50"],
        enriched["atr14"],
    )
    h1 = resample_ohlcv(data, "1h")
    h1["h1_ema50"] = ema(h1["close"], 50)
    h1["h1_slope"] = h1["h1_ema50"].diff()
    h1_context = h1[["datetime", "h1_ema50", "h1_slope"]].rename(columns={"datetime": "h1_time"})
    enriched["h1_time"] = enriched["datetime"].dt.floor("h")
    enriched = enriched.merge(h1_context, on="h1_time", how="left")
    return enriched.drop(columns=["h1_time"])


def apply_filters(
    data: pd.DataFrame,
    session_filter: str,
    trend_filter: str,
    volatility_filter: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=data.index)
    if session_filter == "london":
        mask &= data["hour"].between(7, 11)
    elif session_filter == "ny":
        mask &= data["hour"].between(13, 17)
    elif session_filter == "london_ny":
        mask &= data["hour"].between(7, 17)

    if trend_filter == "above_h1_ema":
        mask &= data["close"] > data["h1_ema50"]
    elif trend_filter == "h1_slope_up":
        mask &= data["h1_slope"] > 0

    atr = data["atr14"]
    if volatility_filter == "atr_above_median":
        mask &= atr > atr.median()
    elif volatility_filter == "atr_middle_80":
        mask &= atr.between(atr.quantile(0.1), atr.quantile(0.9))

    return data.loc[mask].copy().reset_index(drop=True)


def filter_key(row: pd.Series) -> str:
    return f"|{row['session_filter']}|{row['trend_filter']}|{row['volatility_filter']}"


def write_summary(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    results: pd.DataFrame,
    robust: pd.DataFrame,
    simulations: pd.DataFrame,
) -> None:
    top = robust.head(10) if not robust.empty else results.head(10)
    cols = [
        "variant_id",
        "session_filter",
        "trend_filter",
        "volatility_filter",
        "test_trades",
        "test_win_rate",
        "test_profit_factor",
        "test_net_expectancy",
        "test_total_return",
        "test_max_drawdown",
        "train_net_expectancy",
    ]
    lines = [
        "# EURUSD M15 Reversion Refinement",
        "",
        f"- Rows: {len(data):,}",
        f"- Train rows: {len(train):,}",
        f"- Test rows: {len(test):,}",
        f"- Variants tested: {len(results):,}",
        f"- Robust candidates: {len(robust):,}",
        "",
        "## Top Candidates",
        "",
        top[cols].to_markdown(index=False),
        "",
        "## Risk / FTMO Simulations",
        "",
    ]
    if simulations.empty:
        lines.append("No robust candidates passed simulation filters.")
    else:
        lines.append(simulations.head(24).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Practical Interpretation",
            "",
            "- Session and volatility filters are only useful if they improve test expectancy without collapsing trade count.",
            "- Prefer candidates that stay positive with conservative costs.",
            "- A high FTMO pass probability with high fail probability is not acceptable for live challenge planning.",
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
