"""FastAPI application for QuantTech100."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.ai.assistant import get_assistant_status, parse_research_prompt
from backend.config import settings
from backend.core.cost_model import CostConfig
from backend.core.data_loader import DataLoadError, load_csv, load_yahoo_finance
from backend.core.data_validator import validate_ohlcv
from backend.core.montecarlo import run_monte_carlo
from backend.core.prop_firm_simulator import PRESETS, simulate_prop_firm
from backend.core.research_engine import Hypothesis, run_ema_mean_reversion_research


app = FastAPI(title="QuantTech100", version="0.5.0")


class YahooRequest(BaseModel):
    symbol: str
    start: str | None = None
    end: str | None = None
    interval: str = "1d"


class ResearchRequest(BaseModel):
    dataset: str
    asset: str = "UNKNOWN"
    timeframe: str = "1D"
    description: str = "EMA mean reversion research"
    ema_window: int = 100
    atr_window: int = 14
    distance_atr: float = 2.0
    exit_distance_atr: float = 0.25
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01
    spread: float = 0.0
    commission_per_unit: float = 0.0
    slippage: float = 0.0
    fixed_cost: float = 0.0
    variable_rate: float = 0.0
    prop_firm: str = "FTMO"
    monte_carlo_simulations: int = 10_000


class MonteCarloRequest(BaseModel):
    trade_r: list[float]
    simulations: int = 10_000
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01


class PropFirmRequest(BaseModel):
    trade_r: list[float]
    preset: str = "FTMO"
    simulations: int = 10_000
    initial_capital: float = 100_000.0


class AssistantRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("frontend/index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    target = raw_dir / file.filename
    target.write_bytes(await file.read())
    try:
        data = load_csv(target)
    except DataLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"dataset": file.filename, "rows": len(data), "validation": asdict(validate_ohlcv(data))}


@app.post("/load-yahoo")
def load_yahoo(request: YahooRequest) -> dict[str, Any]:
    try:
        data = load_yahoo_finance(
            request.symbol,
            start=request.start,
            end=request.end,
            interval=request.interval,
        )
    except DataLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output = Path(settings.data_dir) / "market_cache" / f"{request.symbol}_{request.interval}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return {"dataset": output.name, "rows": len(data), "path": str(output)}


@app.get("/datasets")
def datasets() -> dict[str, list[str]]:
    data_dir = Path(settings.data_dir)
    files = sorted(str(path.relative_to(data_dir)) for path in data_dir.rglob("*.csv"))
    return {"datasets": files}


@app.post("/run-backtest")
def run_backtest_endpoint(request: ResearchRequest) -> dict[str, Any]:
    dataset_path = _resolve_dataset(request.dataset)
    data = load_csv(dataset_path)
    hypothesis = Hypothesis(
        name="EMA ATR Mean Reversion",
        description=request.description,
        assets=(request.asset,),
        timeframes=(request.timeframe,),
        parameters={
            "ema_window": request.ema_window,
            "atr_window": request.atr_window,
            "distance_atr": request.distance_atr,
            "exit_distance_atr": request.exit_distance_atr,
        },
    )
    summary = run_ema_mean_reversion_research(
        data,
        hypothesis=hypothesis,
        output_root=settings.reports_dir,
        initial_capital=request.initial_capital,
        risk_per_trade=request.risk_per_trade,
        cost_config=CostConfig(
            spread=request.spread,
            commission_per_unit=request.commission_per_unit,
            slippage=request.slippage,
            fixed_cost=request.fixed_cost,
            variable_rate=request.variable_rate,
        ),
        prop_firm=request.prop_firm,
        monte_carlo_simulations=request.monte_carlo_simulations,
    )
    return summary


@app.post("/run-montecarlo")
def montecarlo_endpoint(request: MonteCarloRequest) -> dict[str, Any]:
    result = run_monte_carlo(
        pd.Series(request.trade_r),
        simulations=request.simulations,
        initial_capital=request.initial_capital,
        risk_per_trade=request.risk_per_trade,
    )
    return asdict(result)


@app.post("/run-prop-firm-simulation")
def prop_firm_endpoint(request: PropFirmRequest) -> dict[str, Any]:
    if request.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")
    result = simulate_prop_firm(
        pd.Series(request.trade_r),
        rules=PRESETS[request.preset],
        initial_capital=request.initial_capital,
        simulations=request.simulations,
    )
    return asdict(result)


@app.post("/create-hypothesis")
def create_hypothesis(request: ResearchRequest) -> dict[str, Any]:
    hypothesis = Hypothesis(
        name="EMA ATR Mean Reversion",
        description=request.description,
        assets=(request.asset,),
        timeframes=(request.timeframe,),
        parameters={
            "ema_window": request.ema_window,
            "atr_window": request.atr_window,
            "distance_atr": request.distance_atr,
            "exit_distance_atr": request.exit_distance_atr,
        },
    )
    return asdict(hypothesis)


@app.get("/results/{run_id}")
def get_results(run_id: str) -> dict[str, str]:
    summary = Path(settings.reports_dir) / run_id / "summary.json"
    if not summary.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return {"summary": summary.read_text(encoding="utf-8")}


@app.get("/ai/status")
def ai_status() -> dict[str, Any]:
    return asdict(get_assistant_status())


@app.post("/ai/parse")
def ai_parse(request: AssistantRequest) -> dict[str, object]:
    return parse_research_prompt(request.prompt)


def _resolve_dataset(dataset: str) -> Path:
    data_dir = Path(settings.data_dir)
    candidates = [
        data_dir / dataset,
        data_dir / "raw" / dataset,
        data_dir / "market_cache" / dataset,
        Path(dataset),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset}")
