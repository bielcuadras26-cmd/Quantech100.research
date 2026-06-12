"""Trading cost model for gross-to-net result conversion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Direction(IntEnum):
    """Trade direction expressed as PnL sign."""

    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class CostConfig:
    """Configurable transaction cost assumptions for one asset or venue."""

    spread: float = 0.0
    commission_per_unit: float = 0.0
    slippage: float = 0.0
    fixed_cost: float = 0.0
    variable_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TradeCostInput:
    """Minimal trade result inputs needed to calculate costs."""

    entry_price: float
    exit_price: float
    quantity: float
    direction: Direction
    risk_amount: float

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.exit_price <= 0:
            raise ValueError("exit_price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.risk_amount <= 0:
            raise ValueError("risk_amount must be positive")


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed cost and net result calculation."""

    spread_cost: float
    commission_cost: float
    slippage_cost: float
    fixed_cost: float
    variable_cost: float
    total_cost: float
    gross_result: float
    net_result: float
    gross_r: float
    net_r: float
    cost_r: float


def calculate_trade_costs(trade: TradeCostInput, config: CostConfig) -> CostBreakdown:
    """Calculate gross result, costs, net result, and R multiples."""

    gross_result = (trade.exit_price - trade.entry_price) * trade.quantity * int(trade.direction)

    spread_cost = config.spread * trade.quantity
    commission_cost = config.commission_per_unit * trade.quantity * 2
    slippage_cost = config.slippage * trade.quantity * 2
    fixed_cost = config.fixed_cost
    variable_cost = abs(gross_result) * config.variable_rate
    total_cost = spread_cost + commission_cost + slippage_cost + fixed_cost + variable_cost

    net_result = gross_result - total_cost
    gross_r = gross_result / trade.risk_amount
    net_r = net_result / trade.risk_amount
    cost_r = total_cost / trade.risk_amount

    return CostBreakdown(
        spread_cost=spread_cost,
        commission_cost=commission_cost,
        slippage_cost=slippage_cost,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        total_cost=total_cost,
        gross_result=gross_result,
        net_result=net_result,
        gross_r=gross_r,
        net_r=net_r,
        cost_r=cost_r,
    )
