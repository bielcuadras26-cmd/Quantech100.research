"""EURUSD M5 mean-reversion research study.

This script is intentionally reproducible and file-based: it reads an MT5 CSV export,
runs a parameter grid, writes ranked results, and prints an interpretable summary.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from backend.core.cost_model import CostConfig, Direction, TradeCostInput, calculate_trade_costs
from backend.core.data_loader import load_mt5_export
from backend.core.data_validator import validate_ohlcv
from backend.core.metrics import calculate_metrics
from backend.core.montecarlo import run_monte_carlo
from backend.core.prop_firm_simulator import PRESETS, simulate_prop_firm
from backend.strategies.ema_mean_reversion import generate_signals


SOURCE = Path("/Users/bielcuadras/Documents/barras baktest/EURUSD_M5_202501070115_202605111510.csv")
OUTPUT_DIR = Path("research/results/eurusd_m5_mean_reversion")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_mt5_export(SOURCE)
    report = validate_ohlcv(data, expected_frequency="5min")
    data.to_csv(OUTPUT_DIR / "canonical_eurusd_m5.csv", index=False)

    split_index = int(len(data) * 0.7)
    train = data.iloc[:split_index].reset_index(drop=True)
    test = data.iloc[split_index:].reset_index(drop=True)

    cost_cases = {
        "conservative": CostConfig(spread=0.00002, commission_per_unit=0.000035, slippage=0.00001),
    }

    rows: list[dict[str, object]] = []
    trade_books: dict[str, pd.DataFrame] = {}
    for cost_name, costs in cost_cases.items():
        for side in ["long_only", "short_only"]:
            for ema_window in [50, 100]:
                for distance_atr in [1.5, 2.0]:
                    for stop_atr in [1.0, 1.5]:
                        for take_profit_r in [1.0, 2.0]:
                            for max_bars in [12, 24]:
                                params = {
                                    "ema_window": ema_window,
                                    "atr_window": 14,
                                    "distance_atr": distance_atr,
                                    "exit_distance_atr": 0.25,
                                    "stop_atr": stop_atr,
                                    "take_profit_r": take_profit_r,
                                    "max_bars": max_bars,
                                    "side": side,
                                    "cost_case": cost_name,
                                }
                                row, trades = evaluate_variant(train, test, params, costs)
                                rows.append(row)
                                if trades is not None:
                                    trade_books[str(row["variant_id"])] = trades

    results = pd.DataFrame(rows).sort_values(
        ["test_net_expectancy", "test_profit_factor", "test_trades"],
        ascending=[False, False, False],
    )
    results.to_csv(OUTPUT_DIR / "variant_results.csv", index=False)

    robust = results[
        (results["train_trades"] >= 30)
        & (results["test_trades"] >= 15)
        & (results["train_net_expectancy"] > 0)
        & (results["test_net_expectancy"] > 0)
        & (results["test_max_drawdown"] > -0.08)
    ].copy()
    robust.to_csv(OUTPUT_DIR / "robust_candidates.csv", index=False)

    top = robust.head(10) if not robust.empty else results.head(10)
    simulations = []
    for _, row in top.iterrows():
        trades = trade_books.get(str(row["variant_id"]))
        if trades is None or trades.empty:
            continue
        for risk in [0.0025, 0.005, 0.01]:
            mc = run_monte_carlo(
                trades["net_r"],
                simulations=2_000,
                initial_capital=100_000,
                risk_per_trade=risk,
                target_return=0.10,
                max_drawdown_limit=0.10,
            )
            ftmo_rules = replace(PRESETS["FTMO"], risk_per_trade=risk)
            prop = simulate_prop_firm(
                trades["net_r"],
                rules=ftmo_rules,
                initial_capital=100_000,
                simulations=2_000,
            )
            simulations.append(
                {
                    "variant_id": row["variant_id"],
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
    simulation_results.to_csv(OUTPUT_DIR / "risk_and_ftmo_simulations.csv", index=False)

    write_markdown_summary(report, data, train, test, results, robust, simulation_results)


def evaluate_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    params: dict[str, object],
    costs: CostConfig,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    train_result = run_one(train, params, costs)
    test_result = run_one(test, params, costs)
    variant_id = (
        f"{params['cost_case']}|{params['side']}|ema{params['ema_window']}|"
        f"d{params['distance_atr']}|sl{params['stop_atr']}|tp{params['take_profit_r']}|mb{params['max_bars']}"
    )

    row = {
        "variant_id": variant_id,
        **params,
        **prefix_metrics("train", train_result["metrics"]),
        **prefix_metrics("test", test_result["metrics"]),
    }
    return row, test_result["trades"]


def run_one(data: pd.DataFrame, params: dict[str, object], costs: CostConfig) -> dict[str, object]:
    enriched, signals = generate_signals(
        data,
        ema_window=int(params["ema_window"]),
        atr_window=int(params["atr_window"]),
        distance_atr=float(params["distance_atr"]),
        exit_distance_atr=float(params["exit_distance_atr"]),
    )
    side = str(params["side"])
    if side == "long_only":
        signals["short_entry"] = False
    elif side == "short_only":
        signals["long_entry"] = False

    trades, equity_curve = fast_backtest(
        enriched,
        signals,
        stop_atr=float(params["stop_atr"]),
        take_profit_r=float(params["take_profit_r"]),
        max_bars=int(params["max_bars"]),
        costs=costs,
    )
    metrics = calculate_metrics(trades, equity_curve, initial_capital=100_000)
    return {"metrics": asdict(metrics), "trades": trades}


def fast_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    stop_atr: float,
    take_profit_r: float,
    max_bars: int,
    costs: CostConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = data["datetime"].to_numpy()
    high = data["high"].to_numpy(dtype="float64")
    low = data["low"].to_numpy(dtype="float64")
    close = data["close"].to_numpy(dtype="float64")
    atr_values = data["atr"].to_numpy(dtype="float64")
    long_entries = signals["long_entry"].to_numpy(dtype="bool")
    short_entries = signals["short_entry"].to_numpy(dtype="bool")
    exits = signals["exit"].to_numpy(dtype="bool")

    capital = 100_000.0
    risk_fraction = 0.005
    equity = []
    trades = []
    open_trade = None

    for index in range(len(data)):
        if open_trade is not None:
            (
                entry_index,
                entry_time,
                direction,
                entry_price,
                stop_loss,
                take_profit,
                quantity,
                risk_amount,
                mae,
                mfe,
            ) = open_trade
            denominator = abs(entry_price - stop_loss)
            if direction == Direction.LONG:
                mae = min(mae, (low[index] - entry_price) / denominator)
                mfe = max(mfe, (high[index] - entry_price) / denominator)
                exit_price = None
                reason = None
                if low[index] <= stop_loss:
                    exit_price = stop_loss
                    reason = "stop_loss"
                elif high[index] >= take_profit:
                    exit_price = take_profit
                    reason = "take_profit"
            else:
                mae = min(mae, (entry_price - high[index]) / denominator)
                mfe = max(mfe, (entry_price - low[index]) / denominator)
                exit_price = None
                reason = None
                if high[index] >= stop_loss:
                    exit_price = stop_loss
                    reason = "stop_loss"
                elif low[index] <= take_profit:
                    exit_price = take_profit
                    reason = "take_profit"

            if exit_price is None and exits[index]:
                exit_price = close[index]
                reason = "signal_exit"
            if exit_price is None and index - entry_index >= max_bars:
                exit_price = close[index]
                reason = "time_exit"

            if exit_price is not None:
                breakdown = calculate_trade_costs(
                    TradeCostInput(
                        entry_price=entry_price,
                        exit_price=float(exit_price),
                        quantity=quantity,
                        direction=direction,
                        risk_amount=risk_amount,
                    ),
                    costs,
                )
                capital += breakdown.net_result
                trades.append(
                    {
                        "entry_time": pd.Timestamp(entry_time),
                        "exit_time": pd.Timestamp(timestamps[index]),
                        "direction": direction.name,
                        "entry_price": entry_price,
                        "exit_price": float(exit_price),
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "quantity": quantity,
                        "gross_result": breakdown.gross_result,
                        "net_result": breakdown.net_result,
                        "gross_r": breakdown.gross_r,
                        "net_r": breakdown.net_r,
                        "mae": mae,
                        "mfe": mfe,
                        "costs": breakdown.total_cost,
                        "exit_reason": reason,
                    }
                )
                open_trade = None
            else:
                open_trade = (
                    entry_index,
                    entry_time,
                    direction,
                    entry_price,
                    stop_loss,
                    take_profit,
                    quantity,
                    risk_amount,
                    mae,
                    mfe,
                )

        if open_trade is None and not pd.isna(atr_values[index]) and atr_values[index] > 0:
            direction = None
            if long_entries[index]:
                direction = Direction.LONG
            elif short_entries[index]:
                direction = Direction.SHORT
            if direction is not None:
                entry_price = float(close[index])
                stop_distance = float(atr_values[index]) * stop_atr
                risk_amount = capital * risk_fraction
                quantity = risk_amount / stop_distance
                if direction == Direction.LONG:
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + stop_distance * take_profit_r
                else:
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - stop_distance * take_profit_r
                open_trade = (
                    index,
                    timestamps[index],
                    direction,
                    entry_price,
                    stop_loss,
                    take_profit,
                    quantity,
                    risk_amount,
                    0.0,
                    0.0,
                )

        equity.append({"datetime": timestamps[index], "equity": capital})

    return pd.DataFrame(trades), pd.DataFrame(equity)


def prefix_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    selected = {
        "trades": metrics["number_of_trades"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["net_profit_factor"],
        "net_expectancy": metrics["net_expectancy"],
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe_ratio"],
        "sortino": metrics["sortino_ratio"],
        "recovery_factor": metrics["recovery_factor"],
    }
    return {f"{prefix}_{key}": value for key, value in selected.items()}


def write_markdown_summary(
    validation_report,
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    results: pd.DataFrame,
    robust: pd.DataFrame,
    simulations: pd.DataFrame,
) -> None:
    top = robust.head(5) if not robust.empty else results.head(5)
    lines = [
        "# EURUSD M5 Mean Reversion Study",
        "",
        "## Dataset",
        "",
        f"- Rows: {len(data):,}",
        f"- Start: {data['datetime'].iloc[0]}",
        f"- End: {data['datetime'].iloc[-1]}",
        f"- Train rows: {len(train):,}",
        f"- Test rows: {len(test):,}",
        f"- Validation valid: {validation_report.is_valid}",
        f"- Quality score: {validation_report.quality_score}",
        f"- Warnings: {[warning.code for warning in validation_report.warnings]}",
        "",
        "## Search Space",
        "",
        "- EMA: 50, 100",
        "- ATR window: 14",
        "- Distance: 1.5, 2.0 ATR",
        "- Stop: 1.0, 1.5 ATR",
        "- TP: 1.0, 2.0 R",
        "- Max bars: 12, 24",
        "- Side: long only, short only",
        "- Costs: conservative",
        "",
        "## Top Robust Candidates",
        "",
    ]
    display_cols = [
        "variant_id",
        "test_trades",
        "test_win_rate",
        "test_profit_factor",
        "test_net_expectancy",
        "test_total_return",
        "test_max_drawdown",
        "train_net_expectancy",
    ]
    lines.append(top[display_cols].to_markdown(index=False))
    lines.extend(["", "## FTMO / Risk Simulations", ""])
    if simulations.empty:
        lines.append("No simulation candidates met the minimum robustness filters.")
    else:
        lines.append(simulations.head(15).to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Prefer candidates with positive train and test expectancy.",
            "- Ignore variants with very few trades.",
            "- Treat FTMO pass probability as a stress estimate, not a promise.",
            "- If Backtesting.py or VectorBT disagree later, inspect execution assumptions before trusting either result.",
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
