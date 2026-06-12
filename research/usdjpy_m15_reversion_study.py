"""USDJPY M15 mean-reversion study with JPY-to-USD PnL conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from backend.core.cost_model import CostConfig, Direction
from backend.core.data_loader import load_mt5_export
from backend.core.metrics import calculate_metrics
from backend.core.prop_firm_simulator import PRESETS
from backend.strategies.ema_mean_reversion import generate_signals
from research.eurusd_m15_reversion_refinement import add_context, resample_ohlcv
from research.eurusd_m15_walk_forward_funding import build_quarterly_folds
from research.multi_asset_portfolio_funding import build_asset_calendar, simulate_portfolio_funding


SOURCE = Path("/Users/bielcuadras/Documents/barras baktest/USDJPY_M5_202502070200_202606121735.csv")
OUTPUT_DIR = Path("research/results/usdjpy_m15_reversion")
INITIAL_CAPITAL = 100_000.0
POINT = 0.001
COMMISSION_PER_UNIT_USD = 0.000035

BASE_PARAMS = {
    "ema_window": 50,
    "atr_window": 14,
    "distance_atr": 2.25,
    "exit_distance_atr": 0.25,
    "stop_atr": 2.0,
    "take_profit_r": 2.0,
    "max_bars": 32,
    "side": "trend_aligned",
}

CANDIDATE_PARAMS = {
    "ema_window": 50,
    "atr_window": 14,
    "distance_atr": 2.5,
    "exit_distance_atr": 0.25,
    "stop_atr": 2.5,
    "take_profit_r": 2.0,
    "max_bars": 24,
    "side": "short_only",
}

ROBUSTNESS_GRID = [
    {
        "distance_atr": distance,
        "stop_atr": stop,
        "take_profit_r": tp,
        "max_bars": max_bars,
        "side": side,
    }
    for side in ["long_only", "short_only", "trend_aligned"]
    for distance in [1.75, 2.0, 2.25, 2.5]
    for stop in [1.5, 2.0, 2.5]
    for tp in [1.5, 2.0]
    for max_bars in [24, 32, 48]
]


@dataclass(frozen=True)
class ConvertedCostBreakdown:
    gross_result: float
    net_result: float
    gross_r: float
    net_r: float
    total_cost: float


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, sep="\t")
    spread_points = raw["<SPREAD>"].astype(float)
    spread_p95 = float(spread_points.quantile(0.95)) * POINT
    spread_p99 = float(spread_points.quantile(0.99)) * POINT
    data = resample_ohlcv(load_mt5_export(SOURCE), "15min")
    data = add_context(data)
    data.to_csv(OUTPUT_DIR / "usdjpy_m15_context.csv", index=False)
    folds = build_quarterly_folds(data, train_months=6, test_months=3)

    scenarios = {
        "p95_realistic": CostConfig(
            spread=spread_p95,
            commission_per_unit=COMMISSION_PER_UNIT_USD,
            slippage=spread_p95 * 0.33,
        ),
        "stress": CostConfig(
            spread=max(spread_p99, spread_p95 * 1.5),
            commission_per_unit=COMMISSION_PER_UNIT_USD,
            slippage=spread_p95 * 0.75,
        ),
    }

    summary_rows = []
    fold_rows = []
    funding_rows = []
    robustness_rows = []

    for scenario, costs in scenarios.items():
        base_trades, rows = run_walk_forward(data, folds, BASE_PARAMS, costs)
        fold_rows.extend({"scenario": scenario, **row} for row in rows)
        if not base_trades.empty:
            base_trades["scenario"] = scenario
            base_trades.to_csv(OUTPUT_DIR / f"usdjpy_{scenario}_base_trades.csv", index=False)
            metrics = calculate_metrics(
                base_trades,
                pd.DataFrame(
                    {
                        "datetime": base_trades["exit_time"],
                        "equity": INITIAL_CAPITAL + base_trades["net_result"].cumsum(),
                    }
                ),
                initial_capital=INITIAL_CAPITAL,
            )
            summary_rows.append(
                {
                    "asset": "USDJPY",
                    "scenario": scenario,
                    "variant": "base_trend_aligned",
                    "folds": len(folds),
                    "trades": len(base_trades),
                    "expectancy": metrics.net_expectancy,
                    "profit_factor": metrics.net_profit_factor,
                    "win_rate": metrics.win_rate,
                    "total_return": metrics.total_return,
                    "max_drawdown": metrics.max_drawdown,
                    "spread_p95": spread_p95,
                    "spread_stress": costs.spread,
                }
            )
            base_trades["entry_time"] = pd.to_datetime(base_trades["entry_time"])
            base_trades["exit_time"] = pd.to_datetime(base_trades["exit_time"])
            base_trades["trade_date"] = base_trades["entry_time"].dt.date
            for risk in [0.0025, 0.005, 0.0075]:
                result = simulate_portfolio_funding(
                    {"USDJPY": build_asset_calendar(base_trades)},
                    portfolio="USDJPY",
                    assets=["USDJPY"],
                    scenario=scenario,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    risk=risk,
                    simulations=2_000,
                )
                funding_rows.append({"asset": "USDJPY", **result.__dict__})

        for variant_index, partial in enumerate(ROBUSTNESS_GRID, start=1):
            params = {**BASE_PARAMS, **partial}
            trades, rows = run_walk_forward(data, folds, params, costs)
            robustness_rows.append(summarize_variant(scenario, variant_index, params, trades, rows))

        candidate_trades, candidate_rows = run_walk_forward(data, folds, CANDIDATE_PARAMS, costs)
        fold_rows.extend({"scenario": scenario, "candidate": "short_only_d2.5_sl2.5_tp2", **row} for row in candidate_rows)
        if not candidate_trades.empty:
            candidate_trades["scenario"] = scenario
            candidate_trades.to_csv(OUTPUT_DIR / f"usdjpy_{scenario}_candidate_short_trades.csv", index=False)
            candidate_metrics = calculate_metrics(
                candidate_trades,
                pd.DataFrame(
                    {
                        "datetime": candidate_trades["exit_time"],
                        "equity": INITIAL_CAPITAL + candidate_trades["net_result"].cumsum(),
                    }
                ),
                initial_capital=INITIAL_CAPITAL,
            )
            summary_rows.append(
                {
                    "asset": "USDJPY",
                    "scenario": scenario,
                    "variant": "candidate_short_only_d2.5_sl2.5_tp2",
                    "folds": len(folds),
                    "trades": len(candidate_trades),
                    "expectancy": candidate_metrics.net_expectancy,
                    "profit_factor": candidate_metrics.net_profit_factor,
                    "win_rate": candidate_metrics.win_rate,
                    "total_return": candidate_metrics.total_return,
                    "max_drawdown": candidate_metrics.max_drawdown,
                    "spread_p95": spread_p95,
                    "spread_stress": costs.spread,
                }
            )
            candidate_trades["entry_time"] = pd.to_datetime(candidate_trades["entry_time"])
            candidate_trades["exit_time"] = pd.to_datetime(candidate_trades["exit_time"])
            candidate_trades["trade_date"] = candidate_trades["entry_time"].dt.date
            for risk in [0.0025, 0.005, 0.0075]:
                result = simulate_portfolio_funding(
                    {"USDJPY": build_asset_calendar(candidate_trades)},
                    portfolio="USDJPY_candidate_short",
                    assets=["USDJPY"],
                    scenario=scenario,
                    rules=replace(PRESETS["FTMO"], risk_per_trade=risk),
                    risk=risk,
                    simulations=2_000,
                )
                funding_rows.append({"asset": "USDJPY", **result.__dict__})

    summary = pd.DataFrame(summary_rows)
    fold_results = pd.DataFrame(fold_rows)
    funding = pd.DataFrame(funding_rows)
    robustness = pd.DataFrame(robustness_rows)
    summary.to_csv(OUTPUT_DIR / "asset_summary.csv", index=False)
    fold_results.to_csv(OUTPUT_DIR / "base_walk_forward_folds.csv", index=False)
    funding.to_csv(OUTPUT_DIR / "funding_simulations.csv", index=False)
    robustness.to_csv(OUTPUT_DIR / "robustness_grid.csv", index=False)
    write_summary(summary, fold_results, funding, robustness)


def run_walk_forward(
    data: pd.DataFrame,
    folds: list[dict[str, pd.Timestamp | str]],
    params: dict[str, float | int | str],
    costs: CostConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    enriched, signals = generate_signals(
        data,
        ema_window=int(params["ema_window"]),
        atr_window=int(params["atr_window"]),
        distance_atr=float(params["distance_atr"]),
        exit_distance_atr=float(params["exit_distance_atr"]),
    )
    side = str(params["side"])
    if side == "long_only":
        signals["long_entry"] = signals["long_entry"] & (data["h1_slope"] > 0)
        signals["short_entry"] = False
    elif side == "short_only":
        signals["long_entry"] = False
        signals["short_entry"] = signals["short_entry"] & (data["h1_slope"] < 0)
    elif side == "trend_aligned":
        signals["long_entry"] = signals["long_entry"] & (data["h1_slope"] > 0)
        signals["short_entry"] = signals["short_entry"] & (data["h1_slope"] < 0)
    else:
        raise ValueError(f"Unsupported side: {side}")

    trade_books = []
    rows = []
    for fold in folds:
        mask = (data["datetime"] >= fold["test_start"]) & (data["datetime"] < fold["test_end"])
        fold_data = enriched.loc[mask].reset_index(drop=True)
        fold_signals = signals.loc[mask].reset_index(drop=True)
        if len(fold_data) < 100:
            continue
        trades, equity = fast_backtest_usdjpy(
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
                "asset": "USDJPY",
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
            trades["asset"] = "USDJPY"
            trades["fold"] = fold["fold"]
            trade_books.append(trades)
    return pd.concat(trade_books, ignore_index=True) if trade_books else pd.DataFrame(), rows


def fast_backtest_usdjpy(
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

    capital = INITIAL_CAPITAL
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
                breakdown = calculate_usdjpy_trade(
                    entry_price=entry_price,
                    exit_price=float(exit_price),
                    quantity=quantity,
                    direction=direction,
                    risk_amount=risk_amount,
                    costs=costs,
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
                quantity = risk_amount * entry_price / stop_distance
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


def calculate_usdjpy_trade(
    *,
    entry_price: float,
    exit_price: float,
    quantity: float,
    direction: Direction,
    risk_amount: float,
    costs: CostConfig,
) -> ConvertedCostBreakdown:
    gross_jpy = (exit_price - entry_price) * quantity * int(direction)
    gross_usd = gross_jpy / exit_price
    spread_usd = costs.spread * quantity / exit_price
    slippage_usd = costs.slippage * quantity * 2 / exit_price
    commission_usd = costs.commission_per_unit * quantity * 2
    fixed_usd = costs.fixed_cost
    variable_usd = abs(gross_usd) * costs.variable_rate
    total_cost = spread_usd + slippage_usd + commission_usd + fixed_usd + variable_usd
    net_usd = gross_usd - total_cost
    return ConvertedCostBreakdown(
        gross_result=gross_usd,
        net_result=net_usd,
        gross_r=gross_usd / risk_amount,
        net_r=net_usd / risk_amount,
        total_cost=total_cost,
    )


def summarize_variant(
    scenario: str,
    variant_index: int,
    params: dict[str, float | int | str],
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
        "asset": "USDJPY",
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
        "# USDJPY M15 Reversion Study",
        "",
        "USDJPY PnL and spread are converted from JPY to USD before calculating R multiples.",
        "",
        "Base rules:",
        "",
        "- M15 EMA50 mean reversion",
        "- Trend aligned: long entries only when H1 EMA50 slope is positive; short entries only when H1 slope is negative",
        "- Entry distance 2.25 ATR",
        "- SL 2 ATR, TP 2R, max 32 M15 bars",
        "- Costs from MT5 spread p95 and p99 stress plus $7 per 100k round-turn commission approximation",
        "",
        "## Base Summary",
        "",
    ]
    lines.append(summary.sort_values(["scenario", "expectancy"], ascending=[True, False]).to_markdown(index=False) if not summary.empty else "No base trades.")
    lines.extend(["", "## Funding Simulation", ""])
    lines.append(funding.sort_values(["scenario", "risk"]).to_markdown(index=False) if not funding.empty else "No funding rows.")
    lines.extend(["", "## Walk-Forward Folds", ""])
    lines.append(folds.to_markdown(index=False) if not folds.empty else "No folds.")
    lines.extend(["", "## Top Robustness Variants", ""])
    if robustness.empty:
        lines.append("No robustness rows.")
    else:
        top = robustness[
            (robustness["trades"] >= 20)
            & (robustness["positive_folds"] >= (robustness["folds"] - 1))
            & (robustness["expectancy"] > 0)
        ].sort_values(["scenario", "expectancy"], ascending=[True, False])
        lines.append(top.groupby("scenario", as_index=False).head(12).to_markdown(index=False))
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
