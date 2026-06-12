"""Generic event-driven backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd

from backend.core.cost_model import CostConfig, Direction, TradeCostInput, calculate_trade_costs


class ExitReason(StrEnum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SIGNAL_EXIT = "signal_exit"
    TIME_EXIT = "time_exit"
    END_OF_DATA = "end_of_data"


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    default_stop_atr: float = 2.0
    default_take_profit_r: float = 2.0
    max_bars_in_trade: int | None = None

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        if self.default_stop_atr <= 0:
            raise ValueError("default_stop_atr must be positive")
        if self.default_take_profit_r <= 0:
            raise ValueError("default_take_profit_r must be positive")


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: Direction
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    gross_result: float
    net_result: float
    gross_r: float
    net_r: float
    mae: float
    mfe: float
    costs: float
    exit_reason: ExitReason


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    equity_curve: pd.DataFrame
    final_capital: float

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(trade) for trade in self.trades])


@dataclass
class _OpenTrade:
    entry_index: int
    entry_time: pd.Timestamp
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    mae: float = 0.0
    mfe: float = 0.0


def run_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    costs: CostConfig | None = None,
) -> BacktestResult:
    """Run a one-position-at-a-time backtest from deterministic signals."""

    cfg = config or BacktestConfig()
    cost_cfg = costs or CostConfig()
    market = data.reset_index(drop=True).copy()
    signal_data = signals.reset_index(drop=True).copy()
    _validate_inputs(market, signal_data)

    capital = cfg.initial_capital
    equity_points: list[dict[str, float | pd.Timestamp]] = []
    trades: list[Trade] = []
    open_trade: _OpenTrade | None = None

    for index, row in market.iterrows():
        timestamp = pd.Timestamp(row["datetime"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        signal_row = signal_data.iloc[index]

        if open_trade is not None:
            open_trade.mae = min(open_trade.mae, _favorable_r(open_trade, low, high, adverse=True))
            open_trade.mfe = max(open_trade.mfe, _favorable_r(open_trade, low, high, adverse=False))
            exit_price, reason = _resolve_exit(open_trade, row, signal_row, index, cfg)
            if exit_price is not None and reason is not None:
                trade = _close_trade(open_trade, timestamp, exit_price, reason, cost_cfg)
                trades.append(trade)
                capital += trade.net_result
                open_trade = None

        if open_trade is None:
            direction = _entry_direction(signal_row)
            if direction is not None:
                open_trade = _open_trade(index, timestamp, close, direction, row, capital, cfg)

        equity_points.append({"datetime": timestamp, "equity": capital})

    if open_trade is not None:
        last = market.iloc[-1]
        trade = _close_trade(
            open_trade,
            pd.Timestamp(last["datetime"]),
            float(last["close"]),
            ExitReason.END_OF_DATA,
            cost_cfg,
        )
        trades.append(trade)
        capital += trade.net_result
        equity_points[-1] = {"datetime": pd.Timestamp(last["datetime"]), "equity": capital}

    return BacktestResult(
        trades=tuple(trades),
        equity_curve=pd.DataFrame(equity_points),
        final_capital=capital,
    )


def _validate_inputs(data: pd.DataFrame, signals: pd.DataFrame) -> None:
    required_market = {"datetime", "open", "high", "low", "close"}
    missing_market = required_market - set(data.columns)
    if missing_market:
        raise ValueError(f"market data missing columns: {', '.join(sorted(missing_market))}")
    if len(data) != len(signals):
        raise ValueError("data and signals must have the same length")


def _entry_direction(signal_row: pd.Series) -> Direction | None:
    if bool(signal_row.get("long_entry", False)):
        return Direction.LONG
    if bool(signal_row.get("short_entry", False)):
        return Direction.SHORT
    return None


def _open_trade(
    index: int,
    timestamp: pd.Timestamp,
    entry_price: float,
    direction: Direction,
    row: pd.Series,
    capital: float,
    config: BacktestConfig,
) -> _OpenTrade:
    atr_value = float(row.get("atr", 0.0) or 0.0)
    stop_distance = atr_value * config.default_stop_atr if atr_value > 0 else entry_price * 0.01
    if direction == Direction.LONG:
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + stop_distance * config.default_take_profit_r
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - stop_distance * config.default_take_profit_r

    risk_amount = capital * config.risk_per_trade
    quantity = risk_amount / stop_distance
    return _OpenTrade(
        entry_index=index,
        entry_time=timestamp,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=quantity,
        risk_amount=risk_amount,
    )


def _resolve_exit(
    trade: _OpenTrade,
    row: pd.Series,
    signal_row: pd.Series,
    index: int,
    config: BacktestConfig,
) -> tuple[float | None, ExitReason | None]:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    if trade.direction == Direction.LONG:
        if low <= trade.stop_loss:
            return trade.stop_loss, ExitReason.STOP_LOSS
        if high >= trade.take_profit:
            return trade.take_profit, ExitReason.TAKE_PROFIT
    else:
        if high >= trade.stop_loss:
            return trade.stop_loss, ExitReason.STOP_LOSS
        if low <= trade.take_profit:
            return trade.take_profit, ExitReason.TAKE_PROFIT

    if bool(signal_row.get("exit", False)):
        return close, ExitReason.SIGNAL_EXIT
    if config.max_bars_in_trade is not None and index - trade.entry_index >= config.max_bars_in_trade:
        return close, ExitReason.TIME_EXIT
    return None, None


def _close_trade(
    trade: _OpenTrade,
    exit_time: pd.Timestamp,
    exit_price: float,
    reason: ExitReason,
    costs: CostConfig,
) -> Trade:
    breakdown = calculate_trade_costs(
        TradeCostInput(
            entry_price=trade.entry_price,
            exit_price=exit_price,
            quantity=trade.quantity,
            direction=trade.direction,
            risk_amount=trade.risk_amount,
        ),
        costs,
    )
    return Trade(
        entry_time=trade.entry_time,
        exit_time=exit_time,
        direction=trade.direction,
        entry_price=trade.entry_price,
        exit_price=exit_price,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        quantity=trade.quantity,
        gross_result=breakdown.gross_result,
        net_result=breakdown.net_result,
        gross_r=breakdown.gross_r,
        net_r=breakdown.net_r,
        mae=trade.mae,
        mfe=trade.mfe,
        costs=breakdown.total_cost,
        exit_reason=reason,
    )


def _favorable_r(trade: _OpenTrade, low: float, high: float, *, adverse: bool) -> float:
    denominator = abs(trade.entry_price - trade.stop_loss)
    if trade.direction == Direction.LONG:
        price_move = (low if adverse else high) - trade.entry_price
    else:
        price_move = trade.entry_price - (high if adverse else low)
    return price_move / denominator
