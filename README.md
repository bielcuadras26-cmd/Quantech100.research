# QuantTech100

QuantTech100 is a private quantitative research platform focused on statistically verifiable market research.

It is not a trading signals platform. The priority is mathematical correctness, reproducibility, and clean research workflows before visual polish.

## Scope

Current milestone:

- Phase 0: clean project foundation
- Phase 0.5: Git/GitHub preparation guide
- Phase 1: architecture and research design documentation

Future milestones will add the data engine, indicator engine, cost model, backtesting engine, Monte Carlo simulations, prop firm simulations, API, and frontend only after the mathematical core is validated.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.main
pytest
```
