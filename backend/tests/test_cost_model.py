import pytest

from backend.core.cost_model import CostConfig, Direction, TradeCostInput, calculate_trade_costs


def test_calculate_trade_costs_for_long_trade() -> None:
    trade = TradeCostInput(
        entry_price=100.0,
        exit_price=110.0,
        quantity=2.0,
        direction=Direction.LONG,
        risk_amount=10.0,
    )
    config = CostConfig(
        spread=0.1,
        commission_per_unit=0.05,
        slippage=0.2,
        fixed_cost=1.0,
        variable_rate=0.01,
    )

    result = calculate_trade_costs(trade, config)

    assert result.gross_result == 20.0
    assert result.spread_cost == 0.2
    assert result.commission_cost == 0.2
    assert result.slippage_cost == 0.8
    assert result.fixed_cost == 1.0
    assert result.variable_cost == 0.2
    assert result.total_cost == pytest.approx(2.4)
    assert result.net_result == pytest.approx(17.6)
    assert result.gross_r == 2.0
    assert result.net_r == pytest.approx(1.76)
    assert result.cost_r == pytest.approx(0.24)


def test_calculate_trade_costs_for_short_trade() -> None:
    trade = TradeCostInput(
        entry_price=100.0,
        exit_price=90.0,
        quantity=1.5,
        direction=Direction.SHORT,
        risk_amount=15.0,
    )
    config = CostConfig(spread=0.2)

    result = calculate_trade_costs(trade, config)

    assert result.gross_result == 15.0
    assert result.total_cost == pytest.approx(0.3)
    assert result.net_result == pytest.approx(14.7)


def test_cost_config_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="spread must be non-negative"):
        CostConfig(spread=-0.1)


def test_trade_cost_input_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="quantity must be positive"):
        TradeCostInput(
            entry_price=100.0,
            exit_price=101.0,
            quantity=0.0,
            direction=Direction.LONG,
            risk_amount=10.0,
        )
