"""TASK 9 (deploy) — FastAPI wrapper around the LangGraph agent.

Endpoints
  GET  /health                     liveness probe
  GET  /tools                      tools discovered via MCP tools/list
  POST /webhook                    send a message           {"message": "...", "thread_id": "optional"}
  POST /webhook/review             HITL approve / reject    {"thread_id": "...", "decision": "approve", "comment": "..."}
  GET  /threads/{thread_id}        inspect persisted state (checkpoint)

Run locally:  uvicorn api.main:app --reload --port 7860   → open http://localhost:7860/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from claimguard.agent import ClaimGuardAgent, ReviewNotPendingError, ThreadAwaitingReviewError
from claimguard.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
agent = ClaimGuardAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.start()          # spawns MCP server, tools/list, opens SQLite checkpointer
    yield
    await agent.stop()


app = FastAPI(title="ClaimGuard AI", version="1.0.0", lifespan=lifespan,
              description="AI-powered insurance claims assessment agent (LangGraph + MCP + HITL).")


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


class WebhookRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, examples=["Assess claim C1003"])
    thread_id: Optional[str] = Field(None, max_length=100, examples=["demo-1"])
    user_id: Optional[str] = Field(None, max_length=100)


class ReviewRequest(BaseModel):
    thread_id: str = Field(..., max_length=100)
    decision: Literal["approve", "reject", "Approve", "Reject"]
    comment: str = Field("", max_length=1000)
    reviewer: str = Field("api-reviewer", max_length=100)


def _public(result: dict, include_events: bool) -> dict:
    out = {k: v for k, v in result.items() if k != "events"}
    if include_events:
        out["events"] = result.get("events", [])
    return out


@app.get("/")
async def root():
    return {"service": "ClaimGuard AI", "docs": "/docs", "tool_mode": agent.provider.mode,
            "tools": len(agent.provider.tools), "telemetry": agent.telemetry.enabled}


@app.get("/health")
async def health():
    return {"status": "ok", "tools_loaded": len(agent.provider.tools)}


@app.get("/tools", dependencies=[Depends(require_api_key)])
async def list_tools():
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in agent.provider.tools]


@app.post("/webhook", dependencies=[Depends(require_api_key)])
async def webhook(req: WebhookRequest, debug: bool = False):
    try:
        result = await agent.chat(req.thread_id, req.message, user_id=req.user_id)
    except ThreadAwaitingReviewError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _public(result, debug)


@app.post("/webhook/review", dependencies=[Depends(require_api_key)])
async def review(req: ReviewRequest, debug: bool = False):
    try:
        result = await agent.resume(req.thread_id, req.decision.lower(), req.comment, req.reviewer)
    except ReviewNotPendingError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _public(result, debug)


@app.get("/threads/{thread_id}", dependencies=[Depends(require_api_key)])
async def thread_state(thread_id: str):
    state = await agent.get_thread(thread_id)
    if not state["exists"]:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found.")
    return state
