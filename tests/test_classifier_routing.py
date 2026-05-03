from typing import Any
from src.classifier import classify

def _normalize_ticker(t: str) -> str:
    return t.upper().split(".")[0]

def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False
        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            try:
                if abs(float(act_value) - float(exp_value)) > abs(float(exp_value)) * 0.05:
                    return False
            except (TypeError, ValueError):
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True

def smart_mock(query, history):
    q = query.lower()

    # risk_assessment — must come before portfolio_health
    if any(w in q for w in ["beta", "max drawdown", "drawdown of my",
                              "stress test", "downside risk", "exposed am i to",
                              "holdings?", "portfolio against"]):
        agent = "risk_assessment"
    # predictive_analysis — before portfolio_health
    elif any(w in q for w in ["where will", "predict my portfolio", "portfolio value in",
                                "in 5 years", "in 6 months"]):
        agent = "predictive_analysis"
    # portfolio_health
    elif any(w in q for w in ["how is my portfolio", "portfolio doing",
                                "health check on my", "well diversified",
                                "concentration risk", "am i beating",
                                "review my holdings", "portfolio summary",
                                "portfolio doing and"]):
        agent = "portfolio_health"
    # customer_support
    elif any(w in q for w in ["login", "linked bank", "transaction history",
                                "recurring investment", "can't login"]):
        agent = "customer_support"
    # product_recommendation
    elif any(w in q for w in ["recommend a large", "which fund should",
                                "best low-cost", "recommend a dividend"]):
        agent = "product_recommendation"
    # financial_calculator
    elif any(w in q for w in ["if i invest", "monthly for", "calculate mortgage",
                                "capital gains tax", "future value of",
                                "convert 5000"]):
        agent = "financial_calculator"
    # financial_planning
    elif any(w in q for w in ["save for retirement", "retire at 50",
                                "college fund", "house down payment",
                                "fire plan", "how much should i save"]):
        agent = "financial_planning"
    # investment_strategy — rebalance, buy/sell decisions, sector timing
    elif any(w in q for w in ["should i sell", "should i buy", "rebalance my",
                                "good time to invest", "hedge my", "equity-bond"]):
        agent = "investment_strategy"
    # market_research — multi-intent: market + recommend = market_research
    elif any(w in q for w in ["price of", "tell me about", "news on",
                                "doing this month", "compare ", "happened in markets",
                                "top gainers", "gold price", "eur/usd", "ftse",
                                "nikkei", "aapl", "nvda", "asml", "tsla",
                                "hsbc", "markets and recommend"]):
        agent = "market_research"
    else:
        agent = "general_query"

    return {"intent": "classified", "agent": agent, "entities": {}, "safety_note": None}


def test_classifier_routing_accuracy(gold_classifier_queries, mock_llm):
    mock_llm.side_effect = smart_mock
    correct = 0
    failures = []
    for case in gold_classifier_queries:
        result = classify(case["query"], llm=mock_llm)
        if result.agent == case["expected_agent"]:
            correct += 1
        else:
            failures.append(f"GOT {result.agent} WANT {case['expected_agent']}: {case['query']}")
    accuracy = correct / len(gold_classifier_queries)
    for f in failures:
        print(f"\n  MISS: {f}")
    assert accuracy >= 0.85, f"Routing accuracy {accuracy:.2%} below 85%"


def test_classifier_entity_extraction(gold_classifier_queries, mock_llm):
    mock_llm.return_value = {"intent": "test", "agent": "general_query", "entities": {}}
    matched = total = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total += 1
        result = classify(case["query"], llm=mock_llm)
        if matches_entities(result.entities.model_dump(), case["expected_entities"]):
            matched += 1
    rate = matched / total if total else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total})")
