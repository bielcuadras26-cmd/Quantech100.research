"""Report generation for research runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backend.core.metrics import calculate_drawdown


def export_research_report(
    *,
    output_dir: str | Path,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, str]:
    """Export trades, summary, equity curve, drawdown curve, and result histogram."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    exports: dict[str, str] = {}

    trades_path = target / "trades.csv"
    trades.to_csv(trades_path, index=False)
    exports["trades_csv"] = str(trades_path)

    summary_path = target / "summary.json"
    summary_path.write_text(_json_text(summary), encoding="utf-8")
    exports["summary_json"] = str(summary_path)

    equity_path = target / "equity_curve.png"
    _plot_line(equity_curve["datetime"], equity_curve["equity"], equity_path, "Equity Curve", "Equity")
    exports["equity_curve"] = str(equity_path)

    drawdown = calculate_drawdown(equity_curve["equity"])
    drawdown_path = target / "drawdown_curve.png"
    _plot_line(equity_curve["datetime"], drawdown, drawdown_path, "Drawdown Curve", "Drawdown")
    exports["drawdown_curve"] = str(drawdown_path)

    if not trades.empty:
        histogram_path = target / "result_histogram.png"
        plt.figure(figsize=(10, 5))
        trades["net_r"].astype(float).hist(bins=30)
        plt.title("Net R Distribution")
        plt.xlabel("Net R")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(histogram_path)
        plt.close()
        exports["result_histogram"] = str(histogram_path)

    return exports


def _plot_line(x_values: pd.Series, y_values: pd.Series, path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _json_text(value: dict[str, Any]) -> str:
    import json

    def default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(cast(Any, obj))
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    return json.dumps(value, indent=2, default=default)
