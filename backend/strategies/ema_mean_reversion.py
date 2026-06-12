"""EMA/ATR mean reversion research strategy."""

from __future__ import annotations

import pandas as pd

from backend.core.indicators import atr, ema, price_to_average_distance_atr


def generate_signals(
    data: pd.DataFrame,
    *,
    ema_window: int = 100,
    atr_window: int = 14,
    distance_atr: float = 2.0,
    exit_distance_atr: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mean-reversion signals when price stretches from EMA by ATR units."""

    enriched = data.copy()
    enriched["ema"] = ema(enriched["close"], ema_window)
    enriched["atr"] = atr(enriched["high"], enriched["low"], enriched["close"], atr_window)
    enriched["distance_atr"] = price_to_average_distance_atr(
        enriched["close"],
        enriched["ema"],
        enriched["atr"],
    )

    signals = pd.DataFrame(index=enriched.index)
    signals["long_entry"] = enriched["distance_atr"] <= -distance_atr
    signals["short_entry"] = enriched["distance_atr"] >= distance_atr
    signals["exit"] = enriched["distance_atr"].abs() <= exit_distance_atr
    signals = signals.fillna(False)

    return enriched, signals
