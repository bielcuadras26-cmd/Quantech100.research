import pandas as pd
from pathlib import Path

from backend.core.montecarlo import run_monte_carlo
from backend.core.prop_firm_simulator import PRESETS, simulate_prop_firm
from backend.core.research_engine import Hypothesis, run_ema_mean_reversion_research
from backend.strategies.ema_mean_reversion import generate_signals


def test_monte_carlo_returns_probabilities() -> None:
    result = run_monte_carlo(pd.Series([1.0, -0.5, 0.75]), simulations=100, seed=1)

    assert result.simulations == 100
    assert 0 <= result.risk_of_ruin <= 1
    assert result.percentile_5 <= result.percentile_95


def test_prop_firm_simulator_returns_probabilities() -> None:
    result = simulate_prop_firm(
        pd.Series([1.0, -0.5, 0.75]),
        rules=PRESETS["FTMO"],
        simulations=100,
        seed=1,
    )

    assert result.simulations == 100
    assert 0 <= result.pass_probability <= 1
    assert 0 <= result.fail_probability <= 1


def test_ema_mean_reversion_generates_signal_columns() -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=140, freq="D"),
            "open": [100.0] * 140,
            "high": [101.0] * 140,
            "low": [99.0] * 140,
            "close": [100.0] * 139 + [96.0],
            "volume": [1000] * 140,
        }
    )

    enriched, signals = generate_signals(data, ema_window=20, atr_window=14, distance_atr=1.0)

    assert "ema" in enriched.columns
    assert "atr" in enriched.columns
    assert {"long_entry", "short_entry", "exit"} <= set(signals.columns)


def test_research_engine_exports_report(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=160, freq="D"),
            "open": [100.0] * 160,
            "high": [101.0] * 160,
            "low": [99.0] * 160,
            "close": [100.0] * 159 + [96.0],
            "volume": [1000] * 160,
        }
    )
    hypothesis = Hypothesis(
        name="Test",
        description="Test hypothesis",
        assets=("TEST",),
        timeframes=("1D",),
        parameters={"ema_window": 20, "atr_window": 14, "distance_atr": 1.0},
    )

    summary = run_ema_mean_reversion_research(
        data,
        hypothesis=hypothesis,
        output_root=tmp_path,
        monte_carlo_simulations=100,
    )

    assert "metrics" in summary
    assert "summary_json" in summary["exports"]
