# QuantTech100 Research Framework

## Research Philosophy

Every result must be traceable to:

- A hypothesis.
- A dataset.
- A strategy configuration.
- A cost model.
- A backtest run.
- A metrics snapshot.
- A Monte Carlo run when applicable.

No conclusion is valid without reproducible inputs and stored outputs.

## Hypothesis Template

Each hypothesis should include:

```text
name
description
created_at
assets
timeframes
parameters
dataset_ids
status
results
conclusion
```

Allowed statuses:

- Pending
- Promising
- Validated
- Discarded

## Research Workflow

1. Define the market behavior being tested.
2. Define assets and timeframes.
3. Define exact entry and exit rules.
4. Define risk assumptions.
5. Define costs before running tests.
6. Validate dataset quality.
7. Run backtest.
8. Calculate metrics.
9. Run Monte Carlo.
10. Run prop firm simulation when relevant.
11. Save all outputs.
12. Write a conclusion.

## Anti-Overfitting Rules

- Avoid judging a hypothesis by one asset or one period.
- Compare in-sample and out-of-sample behavior.
- Track parameter sensitivity.
- Prefer robust parameter zones over single best values.
- Separate exploratory tests from validation tests.
- Include net costs from the beginning.

## Result Interpretation

A hypothesis can be marked as promising when:

- Net expectancy is positive.
- Profit factor is stable after costs.
- Drawdown is compatible with intended risk.
- Monte Carlo percentiles remain acceptable.
- Performance is not concentrated in a tiny subset of trades.

A hypothesis can be marked as validated only after:

- It survives multiple assets or regimes when applicable.
- It has a documented out-of-sample check.
- Its risk profile is acceptable under Monte Carlo simulation.
- Its assumptions are clear enough for independent reproduction.

A hypothesis should be discarded when:

- Net expectancy is negative.
- Costs destroy the edge.
- Drawdown is structurally unacceptable.
- Results depend on narrow overfit parameters.
- Dataset quality is insufficient.

## AI Assistant Rules

The AI research assistant may explain and orchestrate research, but it must never fabricate numerical results.

Required behavior:

- Convert user requests into structured configs.
- Ask for missing dataset assumptions when needed.
- Run backend engines.
- Read stored outputs.
- Explain only what the outputs support.

Forbidden behavior:

- Calculating metrics from memory.
- Inventing trades.
- Inventing Monte Carlo probabilities.
- Ignoring costs.
- Presenting untested hypotheses as validated edges.
