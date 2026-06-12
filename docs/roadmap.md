# QuantTech100 Roadmap

## Phase 0 - Project Foundation

Status: started.

Deliverables:

- Clean repository structure.
- Python virtual environment instructions.
- `requirements.txt`.
- `.gitignore`.
- `README.md`.
- `CHANGELOG.md`.
- `LICENSE`.
- `.env.example`.
- `docs/`.
- `tests/`.
- Smoke verification.

## Phase 0.5 - Git and GitHub

Status: prepared.

Target workflow:

```bash
git init
git branch -M main
git checkout -b develop
git add .
git commit -m "Initial QuantTech100 Architecture"
```

GitHub requirements:

- Create a private repository.
- Add the remote.
- Push `main` and `develop`.
- Keep commits descriptive and scoped.

Example future commits:

- `feat: add data loader`
- `feat: add backtesting engine`
- `feat: add montecarlo simulator`
- `feat: add prop firm simulator`
- `fix: correct drawdown calculation`
- `refactor: optimize indicator engine`

## Phase 1 - Architecture

Status: started.

Deliverables:

- Architecture documentation.
- Research framework documentation.
- API boundary design.
- Module responsibilities.
- Data flow.
- AI assistant rules.
- Future database design.

## Phase 2 - Data Engine

Status: started.

Core tasks:

- CSV loader.
- Yahoo Finance loader.
- MT5 export parser.
- Dataset validation report.
- Canonical OHLCV format.

Implemented foundation:

- Canonical OHLCV schema normalization.
- CSV loader.
- MT5 export loader.
- Yahoo Finance adapter.
- Chronology, datetime, missing value, duplicate, gap, and OHLC consistency validation.

## Phase 3 - Indicator Engine

Status: started.

Core tasks:

- Implement indicators with tests.
- Prevent look-ahead bias.
- Validate warmup behavior.

Implemented foundation:

- SMA, EMA, ATR, RSI, MACD, Bollinger Bands, Z-score, rolling volatility.
- Momentum, returns, rolling correlation, moving-average slope.
- Price-to-average distance in percent and ATR.
- Rolling highs and lows.

## Phase 4 - Cost Model

Status: started.

Core tasks:

- Asset-level cost config.
- Gross-to-net conversion.
- R-based cost reporting.

Implemented foundation:

- Spread, commission, slippage, fixed cost, and variable-rate costs.
- Long and short gross result calculation.
- Monetary total cost, net result, gross R, net R, and cost R.

## Phase 5 - Backtesting Engine

Status: started.

Core tasks:

- Generic signal execution.
- SL, TP, and time exits.
- Trade ledger.
- MAE and MFE.

Implemented foundation:

- One-position-at-a-time event-driven engine.
- Long and short entries, exits, SL, TP, time exits, end-of-data exits.
- MAE, MFE, gross result, net result, gross R, net R, and costs per trade.

## Phase 6 - Metrics Engine

Status: started.

Core tasks:

- Gross and net metrics.
- Drawdown metrics.
- Distribution metrics.
- Time-based breakdowns.

Implemented foundation:

- Trade count, win rate, profit factor, expectancy, return, drawdown, losing streak.
- Sharpe, Sortino, recovery factor, and period breakdown helper.

## Phase 7 - Monte Carlo Engine

Status: started.

Core tasks:

- 10,000 simulations minimum.
- Percentile curves.
- Drawdown estimates.
- Risk of ruin and target probabilities.

Implemented foundation:

- Resampling from real trade R outcomes.
- Final equity percentiles, expected/worst drawdown, ruin, target, and limit breach probabilities.

## Phase 8 - Prop Firm Simulator

Status: started.

Core tasks:

- Rule presets.
- Pass/fail probabilities.
- Expected payout analysis.

Implemented foundation:

- FTMO, Topstep, FundingPips, Alpha Capital, Orion, and FundingNext presets.
- Profit target, daily loss, total loss, risk per trade, and max daily trades.

## Phase 9 - Research Engine

Status: started.

Core tasks:

- Hypothesis lifecycle.
- Run orchestration.
- Result comparison.
- Research archive.

Implemented foundation:

- EMA/ATR mean-reversion research orchestration.
- Backtest, metrics, Monte Carlo, prop firm, and report export pipeline.

## Phase 10 - Report Generator

Status: started.

Core tasks:

- CSV exports.
- Charts.
- Statistical summaries.
- Simulation reports.

Implemented foundation:

- Trades CSV, summary JSON, equity curve, drawdown curve, and Net R histogram.

## Phase 11 - FastAPI

Status: started.

Core tasks:

- Upload datasets.
- List datasets.
- Run backtests.
- Run simulations.
- Fetch results.

Implemented foundation:

- Local API endpoints for upload, Yahoo data, datasets, research runs, Monte Carlo, prop firm simulation, results, and AI status.

## Phase 12 - Frontend

Status: started with a static local UI served by FastAPI.

Target stack:

- Next.js
- Tailwind CSS

Note: the current interface is intentionally lightweight so the local research workflow works immediately.

## Phase 13 - AI Research Assistant

Status: prepared.

Core tasks:

- Natural language to structured research config.
- Backend execution only.
- Result-grounded explanations.

Implemented foundation:

- API key status detection.
- Conservative prompt-to-config parser.
- No invented research results.
