from __future__ import annotations
import json
import os
import uuid
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from src.models import ChatRequest
from src.safety import check as safety_check
from src.classifier import classify
from src.router import route

load_dotenv()

app = FastAPI(
    title="Valura AI",
    description="AI co-investor microservice",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store  {session_id: [{"user": ..., "assistant": ...}]}
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, list[dict[str, str]]] = {}

PIPELINE_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data)}


async def _stream_pipeline(
    query: str,
    session_id: str,
    user_context: dict,
) -> AsyncGenerator[dict, None]:

    history = _SESSIONS.get(session_id, [])

    # 1. Safety guard
    yield _sse("delta", {"text": "Checking query safety..."})
    verdict = safety_check(query)
    if verdict.blocked:
        yield _sse("result", {
            "blocked": True,
            "category": verdict.category,
            "message": verdict.message,
        })
        yield _sse("done", {})
        return

    # 2. Classify
    yield _sse("delta", {"text": "Understanding your query..."})
    classification = classify(query, history=history)

    yield _sse("delta", {
        "text": f"Routing to {classification.agent.replace('_', ' ')}..."
    })

    # 3. Route to agent
    result = route(classification, user_context)

    # 4. Store in session history
    summary = classification.intent or query[:80]
    _SESSIONS.setdefault(session_id, []).append({
        "user": query,
        "assistant": summary,
    })
    # Keep last 10 turns only
    _SESSIONS[session_id] = _SESSIONS[session_id][-10:]

    # 5. Stream final result
    yield _sse("result", {
        "agent": classification.agent,
        "intent": classification.intent,
        "entities": classification.entities.model_dump(exclude_none=True),
        "data": result,
    })
    yield _sse("done", {})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "valura-ai"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    async def generator():
        try:
            async for event in _stream_pipeline(
                query=request.query,
                session_id=session_id,
                user_context=request.user_context,
            ):
                yield event
        except Exception as e:
            yield _sse("error", {"message": f"Pipeline error: {str(e)}"})
            yield _sse("done", {})

    return EventSourceResponse(generator())
