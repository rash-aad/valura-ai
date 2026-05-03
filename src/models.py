from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

class SafetyResult(BaseModel):
    blocked: bool
    category: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Classifier output
# ---------------------------------------------------------------------------

class Entities(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    amount: float | None = None
    rate: float | None = None
    period_years: float | None = None
    frequency: str | None = None
    horizon: str | None = None
    time_period: str | None = None
    index: str | None = None
    action: str | None = None
    goal: str | None = None
    currency: str | None = None


VALID_AGENTS = {
    "portfolio_health",
    "market_research",
    "investment_strategy",
    "financial_planning",
    "financial_calculator",
    "risk_assessment",
    "product_recommendation",
    "predictive_analysis",
    "customer_support",
    "general_query",
}

class ClassifierResult(BaseModel):
    intent: str
    agent: str
    entities: Entities = Field(default_factory=Entities)
    safety_note: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Portfolio Health Agent output
# ---------------------------------------------------------------------------

class ConcentrationRisk(BaseModel):
    top_position_pct: float
    top_3_positions_pct: float
    flag: str  # "low" | "medium" | "high"


class Performance(BaseModel):
    total_return_pct: float
    annualized_return_pct: float


class BenchmarkComparison(BaseModel):
    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class Observation(BaseModel):
    severity: str  # "info" | "warning" | "critical"
    text: str


class PortfolioHealthResult(BaseModel):
    concentration_risk: ConcentrationRisk | None = None
    performance: Performance | None = None
    benchmark_comparison: BenchmarkComparison | None = None
    observations: list[Observation] = Field(default_factory=list)
    disclaimer: str
    is_empty_portfolio: bool = False
    build_guidance: str | None = None


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: str | None = None
    user_context: dict[str, Any] = Field(default_factory=dict)


class SSEEvent(BaseModel):
    event: str  # "delta" | "result" | "error" | "done"
    data: Any
