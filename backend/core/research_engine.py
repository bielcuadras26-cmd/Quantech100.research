"""Research orchestration engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from backend.core.backtester import BacktestConfig, run_backtest
from backend.core.cost_model import CostConfig
from backend.core.metrics import calculate_metrics
from backend.core.montecarlo import run_monte_carlo
from backend.core.prop_firm_simulator import PRESETS, simulate_prop_firm
from backend.core.report_generator import export_research_report
from backend.strategies.ema_mean_reversion import generate_signals


@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str
    assets: tuple[str, ...]
    timeframes: tuple[str, ...]
    parameters: dict[str, Any]
    status: str = "Pending"
    created_at: str = datetime.now(UTC).isoformat()


def run_ema_mean_reversion_research(
    data: pd.DataFrame,
    *,
    hypothesis: Hypothesis,
    output_root: str | Path,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    cost_config: CostConfig | None = None,
    prop_firm: str = "FTMO",
    monte_carlo_simulations: int = 10_000,
) -> dict[str, Any]:
    """Execute a full research pipeline for the default EMA mean-reversion hypothesis."""

    run_id = str(uuid4())
    enriched, signals = generate_signals(data, **hypothesis.parameters)
    backtest = run_backtest(
        enriched,
        signals,
        config=BacktestConfig(initial_capital=initial_capital, risk_per_trade=risk_per_trade),
        costs=cost_config or CostConfig(),
    )
    trades = backtest.trades_frame()
    metrics = calculate_metrics(trades, backtest.equity_curve, initial_capital=initial_capital)
    monte_carlo = None
    prop_result = None
    if not trades.empty:
        monte_carlo = run_monte_carlo(
            trades["net_r"],
            simulations=monte_carlo_simulations,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
        )
        rules = PRESETS[prop_firm]
        prop_result = simulate_prop_firm(
            trades["net_r"],
            rules=rules,
            initial_capital=initial_capital,
            simulations=monte_carlo_simulations,
        )

    summary = {
        "run_id": run_id,
        "hypothesis": asdict(hypothesis),
        "metrics": asdict(metrics),
        "monte_carlo": asdict(monte_carlo) if monte_carlo else None,
        "prop_firm": asdict(prop_result) if prop_result else None,
        "final_capital": backtest.final_capital,
    }
    exports = export_research_report(
        output_dir=Path(output_root) / run_id,
        trades=trades,
        equity_curve=backtest.equity_curve,
        summary=summary,
    )
    summary["exports"] = exports
    return summary
