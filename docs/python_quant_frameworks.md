# Python Quant Frameworks Research

This note records external frameworks that can strengthen QuantTech100 validation and benchmarking.

## Recommended Role By Framework

| Framework | Best Use | QuantTech100 Role |
| --- | --- | --- |
| VectorBT | Fast vectorized research, parameter sweeps, multi-asset experiments | Benchmark and future high-speed optimization layer |
| Backtesting.py | Lightweight event/vector backtesting with optimizer | Cross-check simple strategy results |
| Backtrader | Event-driven strategy simulation | Cross-check order lifecycle assumptions |
| bt | Portfolio allocation and rebalancing systems | Future portfolio research layer |
| QuantStats | Portfolio analytics reports | External metric/report validation |
| Empyrical | Risk/performance statistics | External metric validation |
| Zipline Reloaded | Institutional-style pipeline research | Future advanced equities research |
| pandas-ta / TA-Lib | Indicator libraries | Indicator cross-validation |

## Integration Policy

QuantTech100 should keep its own transparent mathematical core, but use mature libraries as independent validators.

Priority order:

1. Keep deterministic internal tests for every formula.
2. Add cross-check tests against external libraries for indicators and metrics.
3. Use VectorBT for high-speed parameter sweeps only after internal semantics are stable.
4. Use Backtesting.py or Backtrader for selected strategy parity tests.
5. Use QuantStats or Empyrical to validate performance statistics.

## Why Not Replace The Core Immediately

External libraries have different assumptions for order timing, commissions, stop execution, adjusted data, cash handling, and leverage. Replacing the core blindly could make results less explainable. The professional approach is to make QuantTech100 transparent first, then compare it against external engines to find discrepancies.

## Next Integration Candidates

- `quantstats`: lowest-risk addition for report/metric cross-checking.
- `empyrical`: useful for Sharpe, Sortino, alpha/beta, drawdown, and returns validation.
- `backtesting.py`: compact cross-check for event-driven strategy behavior.
- `vectorbt`: powerful once parameter grids become large.

## Current Project Integration

Implemented:

- `backtesting.py` adapter for EMA/ATR mean reversion cross-checks.
- `vectorbt` adapter for fast signal-based backtests.
- `quantstats` snapshot for external equity-curve metric validation.

Not implemented:

- `empyrical`, because the latest available package build fails on Python 3.13 due to an outdated build dependency path.
