# QuantTech100 Architecture

## Mission

QuantTech100 is a private quantitative research platform for discovering, validating, and documenting statistical market edges. It is not a signal distribution platform and not an execution terminal.

The core design principle is mathematical correctness before interface design.

## System Overview

```text
Market Data Sources
  -> Data Loader
  -> Data Validator
  -> Processed Dataset Store
  -> Indicator Engine
  -> Strategy / Hypothesis Runner
  -> Cost Model
  -> Backtesting Engine
  -> Metrics Engine
  -> Monte Carlo Engine
  -> Prop Firm Simulator
  -> Report Generator
  -> Research Archive
```

## Backend Modules

### Data Loader

Responsibilities:

- Load local CSV files.
- Normalize MT5 exported files.
- Fetch Yahoo Finance data.
- Prepare adapter boundaries for Polygon and Binance.
- Enforce a canonical OHLCV schema.

Canonical schema:

```text
datetime
open
high
low
close
volume
```

### Data Validator

Responsibilities:

- Validate chronological order.
- Validate datetime parsing and timezone assumptions.
- Detect missing data.
- Detect duplicate timestamps.
- Detect OHLC consistency errors.
- Detect session gaps.
- Produce a dataset quality report before research execution.

### Indicator Engine

Responsibilities:

- Compute deterministic indicators from validated data.
- Avoid look-ahead bias.
- Preserve NaN warmup periods.
- Provide reusable indicator outputs for strategies and hypotheses.

Initial indicators:

- SMA
- EMA
- ATR
- RSI
- MACD
- Bollinger Bands
- Z-score
- Rolling volatility
- Momentum
- Returns
- Correlations
- Moving average slope
- Price-to-average distance in percent
- Price-to-average distance in ATR
- Rolling highs and lows

### Cost Model

Responsibilities:

- Model spread, commission, slippage, fixed costs, and variable costs.
- Support per-asset configuration.
- Calculate gross result, monetary cost, cost in R, and net result.

### Backtesting Engine

Responsibilities:

- Consume deterministic signals.
- Execute entries and exits.
- Apply stop loss, take profit, and time exits.
- Calculate MAE, MFE, gross R, net R, and exit reason for every trade.

Trade record:

```text
entry_time
exit_time
direction
entry_price
exit_price
stop_loss
take_profit
gross_r
net_r
mae
mfe
costs
exit_reason
```

### Metrics Engine

Responsibilities:

- Calculate performance, drawdown, distribution, and time-sliced metrics.
- Separate gross and net results.
- Use validated trade records as the only source of truth.

### Monte Carlo Engine

Responsibilities:

- Run at least 10,000 configurable simulations.
- Resample trade outcomes without inventing performance.
- Estimate equity curve percentiles, drawdowns, risk of ruin, target probability, and limit breach probability.

### Prop Firm Simulator

Responsibilities:

- Simulate account rules over trade distributions.
- Support objective, daily loss, total loss, risk per trade, and max daily trades.
- Provide presets for FTMO, Topstep, FundingPips, Alpha Capital, Orion, and FundingNext.

### Research Engine

Responsibilities:

- Create hypotheses.
- Execute backtests.
- Execute metrics.
- Execute Monte Carlo.
- Execute prop firm simulations.
- Persist results.
- Compare hypotheses across assets, timeframes, and parameter sets.

### Report Generator

Responsibilities:

- Export trades as CSV.
- Export statistical summaries.
- Export equity and drawdown curves.
- Export result histograms.
- Export Monte Carlo and prop firm reports.

## API Boundary

FastAPI will expose:

```text
POST /upload-csv
GET /datasets
POST /run-backtest
POST /run-montecarlo
POST /run-prop-firm-simulation
POST /create-hypothesis
GET /results/{id}
```

The API must call backend engines. It must not duplicate mathematical logic.

## Database Design

Phase 1 keeps persistence file-based for research reproducibility. A later database layer should use PostgreSQL for metadata and object storage or local parquet files for large datasets.

Initial entities:

- Dataset
- Asset
- Timeframe
- Strategy
- Hypothesis
- BacktestRun
- Trade
- MetricsSnapshot
- MonteCarloRun
- PropFirmSimulation
- Report

## AI Research Assistant

The AI assistant is an interface layer, not a calculation engine.

It may:

- Interpret user language.
- Convert requests into structured research parameters.
- Trigger backend runs.
- Read persisted results.
- Explain conclusions.

It must not:

- Invent backtest results.
- Estimate metrics from memory.
- Bypass the backend engines.
- Present unverified claims as research conclusions.

## Frontend Boundary

The frontend is intentionally deferred until the mathematical core is validated.

Planned screens:

- Dashboard
- Data Manager
- Strategy Lab
- Hypothesis Lab
- Monte Carlo Lab
- Prop Firm Lab
- Reports
- AI Chat

## Deployment Boundary

The project should remain VPS-ready:

- Environment-driven settings.
- No hardcoded secrets.
- Deterministic tests.
- Portable file paths.
- Clear dependency lock strategy in a future milestone.
