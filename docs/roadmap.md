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

Core tasks:

- Implement indicators with tests.
- Prevent look-ahead bias.
- Validate warmup behavior.

## Phase 4 - Cost Model

Core tasks:

- Asset-level cost config.
- Gross-to-net conversion.
- R-based cost reporting.

## Phase 5 - Backtesting Engine

Core tasks:

- Generic signal execution.
- SL, TP, and time exits.
- Trade ledger.
- MAE and MFE.

## Phase 6 - Metrics Engine

Core tasks:

- Gross and net metrics.
- Drawdown metrics.
- Distribution metrics.
- Time-based breakdowns.

## Phase 7 - Monte Carlo Engine

Core tasks:

- 10,000 simulations minimum.
- Percentile curves.
- Drawdown estimates.
- Risk of ruin and target probabilities.

## Phase 8 - Prop Firm Simulator

Core tasks:

- Rule presets.
- Pass/fail probabilities.
- Expected payout analysis.

## Phase 9 - Research Engine

Core tasks:

- Hypothesis lifecycle.
- Run orchestration.
- Result comparison.
- Research archive.

## Phase 10 - Report Generator

Core tasks:

- CSV exports.
- Charts.
- Statistical summaries.
- Simulation reports.

## Phase 11 - FastAPI

Core tasks:

- Upload datasets.
- List datasets.
- Run backtests.
- Run simulations.
- Fetch results.

## Phase 12 - Frontend

Deferred until backend correctness is validated.

Target stack:

- Next.js
- Tailwind CSS

## Phase 13 - AI Research Assistant

Core tasks:

- Natural language to structured research config.
- Backend execution only.
- Result-grounded explanations.
