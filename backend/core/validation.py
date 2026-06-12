"""Professional validation workflows for quantitative research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import pandas as pd

from backend.core.backtester import BacktestConfig, run_backtest
from backend.core.cost_model import CostConfig
from backend.core.metrics import MetricsSnapshot, calculate_metrics
from backend.strategies.ema_mean_reversion import generate_signals


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class ValidationFoldResult:
    split: TemporalSplit
    selected_parameters: dict[str, Any]
    train_metrics: MetricsSnapshot | None
    test_metrics: MetricsSnapshot
    train_trades: int
    test_trades: int


@dataclass(frozen=True)
class ValidationSummary:
    folds: tuple[ValidationFoldResult, ...]
    aggregate_test_metrics: dict[str, float]
    stability_score: float


def train_test_split(data: pd.DataFrame, *, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset chronologically into in-sample and out-of-sample sections."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(data) < 2:
        raise ValueError("data must contain at least two rows")

    split_index = int(len(data) * train_ratio)
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def build_walk_forward_splits(
    data: pd.DataFrame,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
) -> tuple[TemporalSplit, ...]:
    """Create rolling walk-forward splits from chronological market data."""

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    if len(data) < train_size + test_size:
        raise ValueError("data is too short for the requested walk-forward split")

    splits: list[TemporalSplit] = []
    start = 0
    fold = 1
    while start + train_size + test_size <= len(data):
        train = data.iloc[start : start + train_size]
        test = data.iloc[start + train_size : start + train_size + test_size]
        splits.append(
            TemporalSplit(
                name=f"fold_{fold}",
                train_start=pd.Timestamp(train.iloc[0]["datetime"]),
                train_end=pd.Timestamp(train.iloc[-1]["datetime"]),
                test_start=pd.Timestamp(test.iloc[0]["datetime"]),
                test_end=pd.Timestamp(test.iloc[-1]["datetime"]),
            )
        )
        start += step
        fold += 1
    return tuple(splits)


def run_walk_forward_validation(
    data: pd.DataFrame,
    *,
    parameter_grid: dict[str, list[Any]],
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    cost_config: CostConfig | None = None,
    selection_metric: str = "net_expectancy",
) -> ValidationSummary:
    """Optimize on each train window and evaluate only on the following test window."""

    splits = build_walk_forward_splits(
        data,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
    )
    folds: list[ValidationFoldResult] = []
    costs = cost_config or CostConfig()

    for split in splits:
        train_data = _slice_by_dates(data, split.train_start, split.train_end)
        test_data = _slice_by_dates(data, split.test_start, split.test_end)
        best_parameters, train_metrics, train_trades = _select_parameters(
            train_data,
            parameter_grid,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            costs=costs,
            selection_metric=selection_metric,
        )
        test_metrics, test_trades = _evaluate_parameters(
            test_data,
            best_parameters,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            costs=costs,
        )
        folds.append(
            ValidationFoldResult(
                split=split,
                selected_parameters=best_parameters,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                train_trades=train_trades,
                test_trades=test_trades,
            )
        )

    return ValidationSummary(
        folds=tuple(folds),
        aggregate_test_metrics=_aggregate_test_metrics(folds),
        stability_score=_stability_score(folds),
    )


def run_out_of_sample_validation(
    data: pd.DataFrame,
    *,
    parameter_grid: dict[str, list[Any]],
    train_ratio: float = 0.7,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.01,
    cost_config: CostConfig | None = None,
    selection_metric: str = "net_expectancy",
) -> ValidationFoldResult:
    """Optimize on one in-sample section and report one untouched out-of-sample result."""

    train_data, test_data = train_test_split(data, train_ratio=train_ratio)
    costs = cost_config or CostConfig()
    best_parameters, train_metrics, train_trades = _select_parameters(
        train_data,
        parameter_grid,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        costs=costs,
        selection_metric=selection_metric,
    )
    test_metrics, test_trades = _evaluate_parameters(
        test_data,
        best_parameters,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        costs=costs,
    )
    split = TemporalSplit(
        name="out_of_sample",
        train_start=pd.Timestamp(train_data.iloc[0]["datetime"]),
        train_end=pd.Timestamp(train_data.iloc[-1]["datetime"]),
        test_start=pd.Timestamp(test_data.iloc[0]["datetime"]),
        test_end=pd.Timestamp(test_data.iloc[-1]["datetime"]),
    )
    return ValidationFoldResult(split, best_parameters, train_metrics, test_metrics, train_trades, test_trades)


def validation_to_dict(summary: ValidationSummary | ValidationFoldResult) -> dict[str, Any]:
    return asdict(summary)


def _select_parameters(
    data: pd.DataFrame,
    parameter_grid: dict[str, list[Any]],
    *,
    initial_capital: float,
    risk_per_trade: float,
    costs: CostConfig,
    selection_metric: str,
) -> tuple[dict[str, Any], MetricsSnapshot, int]:
    best_parameters: dict[str, Any] | None = None
    best_metrics: MetricsSnapshot | None = None
    best_trade_count = 0
    best_score = float("-inf")

    for parameters in _grid(parameter_grid):
        metrics, trade_count = _evaluate_parameters(
            data,
            parameters,
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            costs=costs,
        )
        score = float(getattr(metrics, selection_metric))
        if score > best_score:
            best_score = score
            best_parameters = parameters
            best_metrics = metrics
            best_trade_count = trade_count

    if best_parameters is None or best_metrics is None:
        raise ValueError("parameter_grid produced no parameter combinations")
    return best_parameters, best_metrics, best_trade_count


def _evaluate_parameters(
    data: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    initial_capital: float,
    risk_per_trade: float,
    costs: CostConfig,
) -> tuple[MetricsSnapshot, int]:
    enriched, signals = generate_signals(data, **parameters)
    result = run_backtest(
        enriched,
        signals,
        config=BacktestConfig(initial_capital=initial_capital, risk_per_trade=risk_per_trade),
        costs=costs,
    )
    trades = result.trades_frame()
    metrics = calculate_metrics(trades, result.equity_curve, initial_capital=initial_capital)
    return metrics, len(trades)


def _slice_by_dates(
    data: pd.DataFrame,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> pd.DataFrame:
    dated = data.copy()
    dated["datetime"] = pd.to_datetime(dated["datetime"])
    mask = pd.Series(True, index=dated.index)
    if start is not None:
        mask &= dated["datetime"] >= start
    if end is not None:
        mask &= dated["datetime"] <= end
    return dated.loc[mask].copy().reset_index(drop=True)


def _grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def _aggregate_test_metrics(folds: list[ValidationFoldResult]) -> dict[str, float]:
    if not folds:
        return {}
    return {
        "folds": float(len(folds)),
        "average_net_expectancy": float(
            sum(fold.test_metrics.net_expectancy for fold in folds) / len(folds)
        ),
        "average_win_rate": float(sum(fold.test_metrics.win_rate for fold in folds) / len(folds)),
        "average_max_drawdown": float(
            sum(fold.test_metrics.max_drawdown for fold in folds) / len(folds)
        ),
        "total_test_trades": float(sum(fold.test_trades for fold in folds)),
    }


def _stability_score(folds: list[ValidationFoldResult]) -> float:
    if not folds:
        return 0.0
    positive = sum(1 for fold in folds if fold.test_metrics.net_expectancy > 0)
    traded = sum(1 for fold in folds if fold.test_trades > 0)
    return round((positive / len(folds)) * (traded / len(folds)), 4)
