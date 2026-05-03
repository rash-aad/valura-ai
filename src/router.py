from __future__ import annotations
from typing import Any
from src.models import ClassifierResult
from src.agents.portfolio_health import run as portfolio_health_run

def route(result: ClassifierResult, user: dict, llm: Any = None) -> dict:
    agent = result.agent

    if agent == "portfolio_health":
        return portfolio_health_run(user, llm=llm)

    # All other agents — structured stub
    return {
        "agent": agent,
        "intent": result.intent,
        "entities": result.entities.model_dump(exclude_none=True),
        "status": "not_implemented",
        "message": (
            f"The '{agent}' agent is not yet implemented in this build. "
            f"Your query has been classified correctly and would be handled "
            f"by the {agent.replace('_', ' ')} specialist in the full system."
        ),
        "disclaimer": (
            "This is not investment advice. Valura AI is under active development."
        ),
    }
