from __future__ import annotations
import json
import os
import re
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.models import ClassifierResult, Entities, VALID_AGENTS

load_dotenv()

_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

_SYSTEM = """You are an intent classifier for a financial AI platform.
Analyze the user query and conversation history, then return ONLY a JSON object.

Valid agents:
- portfolio_health      : user asking about their portfolio status, diversification, health
- market_research       : prices, news, market activity, specific tickers, indices
- investment_strategy   : buy/sell decisions, rebalancing, sector allocation
- financial_planning    : retirement, education, house, FIRE, long-term goals
- financial_calculator  : compound interest, mortgage, FV/PV, tax calculations, currency conversion
- risk_assessment       : beta, drawdown, stress test, downside risk, VaR
- product_recommendation: recommending specific ETFs, funds, instruments
- predictive_analysis   : price prediction, portfolio forecast
- customer_support      : login, account, transaction, technical issues
- general_query         : greetings, definitions, education, gibberish, anything else

Return exactly this JSON shape:
{
  "intent": "short description of what user wants",
  "agent": "one of the valid agents above",
  "entities": {
    "tickers": [],
    "topics": [],
    "sectors": [],
    "amount": null,
    "rate": null,
    "period_years": null,
    "frequency": null,
    "horizon": null,
    "time_period": null,
    "index": null,
    "action": null,
    "goal": null,
    "currency": null
  },
  "safety_note": null
}

Entity rules:
- tickers: uppercase, e.g. ["AAPL", "NVDA", "ASML.AS"]
- action: one of buy/sell/hold/hedge/rebalance
- frequency: one of daily/weekly/monthly/yearly
- horizon: one of 6_months/1_year/2_years/5_years/10_years
- time_period: one of today/this_week/this_month/this_year
- goal: one of retirement/education/house/FIRE/emergency_fund
- index: exact name e.g. "S&P 500", "FTSE 100", "NIKKEI 225"
- amount: numeric value only, no currency symbol
- rate: decimal e.g. 0.08 for 8%
- For follow-up queries, carry over entities from prior turns when the user references them implicitly

Only return valid JSON. No markdown, no explanation, no code fences."""


def _build_prompt(query: str, history: list[dict[str, str]]) -> str:
    parts = []
    if history:
        parts.append("Conversation so far:")
        for turn in history[-4:]:
            parts.append(f"  User: {turn['user']}")
            if turn.get("assistant"):
                parts.append(f"  Assistant: {turn['assistant'][:100]}")
        parts.append("")
    parts.append(f"Current query: {query}")
    return "\n".join(parts)


def _parse_response(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text)


def _fallback(query: str) -> ClassifierResult:
    return ClassifierResult(
        intent="unknown",
        agent="general_query",
        entities=Entities(),
        safety_note="classifier_fallback",
    )


def classify(
    query: str,
    history: list[dict[str, str]] | None = None,
    llm: Any = None,
) -> ClassifierResult:
    history = history or []

    # Test injection
    if llm is not None:
        raw = llm(query, history)
        if isinstance(raw, dict):
            agent = raw.get("agent", "general_query")
            if agent not in VALID_AGENTS:
                agent = "general_query"
            return ClassifierResult(
                intent=raw.get("intent", ""),
                agent=agent,
                entities=Entities(**raw.get("entities", {})),
                safety_note=raw.get("safety_note"),
                raw=raw,
            )
        return _fallback(query)

    # Real Gemini call
    try:
        prompt = _build_prompt(query, history)
        response = _CLIENT.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        raw = _parse_response(response.text)
        agent = raw.get("agent", "general_query")
        if agent not in VALID_AGENTS:
            agent = "general_query"
        entities_data = {
            k: v for k, v in raw.get("entities", {}).items()
            if v is not None and v != [] and v != ""
        }
        return ClassifierResult(
            intent=raw.get("intent", ""),
            agent=agent,
            entities=Entities(**entities_data),
            safety_note=raw.get("safety_note"),
            raw=raw,
        )
    except Exception as e:
        print(f"[classifier] error: {e}")
        return _fallback(query)
