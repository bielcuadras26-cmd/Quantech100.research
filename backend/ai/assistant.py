"""AI research assistant boundary.

This module intentionally refuses to invent results. It only reports whether an API key is configured
and prepares structured requests for backend execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantStatus:
    provider: str | None
    ready: bool
    message: str


def get_assistant_status() -> AssistantStatus:
    if os.getenv("OPENAI_API_KEY"):
        return AssistantStatus("openai", True, "OpenAI API key configured.")
    if os.getenv("ANTHROPIC_API_KEY"):
        return AssistantStatus("anthropic", True, "Anthropic API key configured.")
    return AssistantStatus(
        None,
        False,
        "No AI API key configured. Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env.",
    )


def parse_research_prompt(prompt: str) -> dict[str, object]:
    """Create a conservative default config from a human prompt."""

    text = prompt.lower()
    return {
        "strategy": "ema_mean_reversion",
        "ema_window": 100,
        "atr_window": 14,
        "distance_atr": 2.0 if "2 atr" in text or "2atr" in text else 1.5,
        "prop_firm": "FTMO" if "ftmo" in text else "FTMO",
        "risk_per_trade": 0.005 if "0.5%" in text or "0,5%" in text else 0.01,
    }
