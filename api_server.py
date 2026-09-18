"""
FastAPI wrapper around the dental agent's LangGraph, exposing it as a
simple POST /chat endpoint so it can sit behind a chat widget, a WhatsApp
webhook, or anything else that can make an HTTP request.

This is deliberately a thin wrapper: all the actual agent logic still
lives in dental_agent.py (system prompts, tools, the Supervisor routing,
Google Calendar integration, ...). This file's only job is turning HTTP
requests into graph.invoke() calls and back into HTTP responses.

Run it locally:
    pip install -r requirements.txt
    venv\\Scripts\\python.exe -m uvicorn api_server:app --reload --port 8000

Then, from another terminal (or Postman, or your chat widget's frontend):
    POST http://localhost:8000/chat
    {"session_id": "some-id-you-control", "message": "hi, can I book an appointment?"}

    -> {"reply": "...", "session_id": "some-id-you-control"}

`session_id` is entirely caller-controlled -- it's whatever uniquely
identifies ONE ongoing conversation to whoever is calling this API. For a
browser chat widget, that's typically a random UUID generated once per
visitor and stored in the browser (e.g. sessionStorage) so it survives
page reloads within a session. For a WhatsApp integration, the natural
choice is the patient's WhatsApp phone number, since WhatsApp already
gives you a stable per-conversation identifier. Reuse the same session_id
across a conversation's turns; use a new one to start a fresh conversation
with no memory of a previous one. It maps directly onto LangGraph's
`thread_id`, which is what the MemorySaver checkpointer uses to keep each
conversation's message history separate from every other one.

NOTE on conversation memory persistence: dental_agent.py's build_graph()
uses a Postgres-backed checkpointer (via DATABASE_URL, e.g. a free Neon
database) when it's configured, and only falls back to the in-memory
MemorySaver otherwise. MemorySaver keeps conversation history in this
process's RAM only, which is wiped on every restart/redeploy and doesn't
work at all across multiple instances behind a load balancer (a patient's
follow-up could land on a different instance with no memory of earlier
turns). Set DATABASE_URL before deploying for real use -- see
dental_agent.py's _build_checkpointer() for setup steps.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

import dental_agent as agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dental_agent_api")

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Non-fatal startup checks -- logged loudly so they show up in
    whatever hosting platform's logs, without crashing the whole process
    (which would also break test imports of this module)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY is not set -- every /chat request will fail "
            "once it tries to call the model."
        )
    if isinstance(agent.calendar_client, agent.FakeCalendarClient):
        logger.warning(
            "Running with the in-memory FakeCalendarClient (Google Calendar "
            "is not configured) -- bookings will not persist to a real "
            "calendar. See calendar_client.py for setup instructions."
        )
    if not os.environ.get("DATABASE_URL"):
        logger.warning(
            "DATABASE_URL is not set -- conversation memory is in-process "
            "RAM only (MemorySaver) and will be wiped on every restart, "
            "redeploy, or wake-from-sleep. See dental_agent.py's "
            "_build_checkpointer() for how to set up a free Neon Postgres "
            "database and fix this."
        )
    yield
    # Cleanly close the Postgres connection graph.checkpointer opened (if
    # any -- this is a no-op when running on MemorySaver).
    agent.close_checkpointer()


app = FastAPI(
    title=f"{agent.CLINIC_NAME} -- Agent API",
    description="HTTP wrapper around the LangGraph multi-agent dental receptionist.",
    version="1.0.0",
    lifespan=_lifespan,
)

# CORS: wide open for now, which is fine for local development and for a
# quick demo. Before pointing a real chat widget at a real domain, replace
# "*" with that domain (e.g. ["https://yourclinic.com"]) so random other
# websites can't call your API from a visitor's browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Built once at import time and reused for every request. Building it per
# request would reconstruct the whole LangGraph graph (and re-authenticate
# to Google Calendar) on every single message, which is wasteful. This is
# safe to share across requests/threads: all per-conversation state lives
# in the checkpointer, keyed by thread_id -- not on the graph object
# itself, which is stateless once built.
graph = agent.build_graph()

# Same per-turn safety net as the CLI's main() -- caps how deep a single
# request can recurse through Supervisor <-> specialist hops.
RECURSION_LIMIT = 12


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The patient's message.")
    session_id: str = Field(
        ..., min_length=1,
        description=(
            "Caller-assigned ID for one ongoing conversation -- e.g. a "
            "browser-generated UUID for a chat widget, or the patient's "
            "phone number for a WhatsApp integration."
        ),
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@app.get("/health")
def health():
    """Plain liveness check. Useful for uptime monitors, and for hosting
    platforms (Render, Fly.io, ...) that ping this before/while routing
    real traffic to the instance."""
    return {"status": "ok", "clinic": agent.CLINIC_NAME}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send one patient message and get the agent's reply.

    Defined as a plain `def` (not `async def`) on purpose: FastAPI runs
    synchronous endpoint functions in a background thread pool
    automatically, so the blocking Anthropic/Google Calendar API calls
    inside graph.invoke() don't stall the event loop for other concurrent
    requests. If this were `async def` instead, those blocking calls would
    freeze every other in-flight request until this one finished.
    """
    config = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    started_at = time.perf_counter()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=request.message)], "specialist_hops": 0},
            config=config,
        )
    except GraphRecursionError:
        logger.warning("Recursion limit hit for session_id=%s", request.session_id)
        return ChatResponse(
            reply=(
                "Sorry, I got a bit tangled up processing that. Could you "
                "rephrase or split that into separate questions?"
            ),
            session_id=request.session_id,
        )
    except Exception:
        # Never leak an internal stack trace to a public-facing widget or a
        # WhatsApp webhook -- log it here (where you can actually see it,
        # e.g. in the hosting platform's log viewer) and return a generic,
        # safe message to the caller instead.
        logger.exception("Unhandled error for session_id=%s", request.session_id)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on our end. Please try again shortly.",
        )

    reply_text = agent.last_reply_text(result["messages"])
    latency = time.perf_counter() - started_at
    logger.info("session_id=%s latency=%.2fs", request.session_id, latency)

    return ChatResponse(reply=reply_text, session_id=request.session_id)
