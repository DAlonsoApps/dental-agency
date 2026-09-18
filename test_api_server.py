"""
Tests for the FastAPI wrapper (api_server.py).

Same offline philosophy as test_dental_agent.py: fake LLMs and a fake
calendar client stand in for the real ones, so this whole suite costs
nothing, needs no ANTHROPIC_API_KEY, and never touches the network. Reuses
the exact same fakes from test_dental_agent.py rather than redefining them,
so there's one source of truth for "what a fake router/specialist looks
like" across both test files.

Run with:
    pytest test_api_server.py -v
"""

import os

# Must be set before importing dental_agent/api_server: ChatAnthropic's
# constructor is fine without a real key (it doesn't validate eagerly),
# but setting a placeholder here makes that explicit and keeps this file
# runnable standalone, not just as part of the full `pytest` run.
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-for-offline-tests")

# Force this OFF (not just leave-as-is) before importing api_server, whose
# module-level `graph = agent.build_graph()` runs at import time -- once,
# for the whole test session. If a real DATABASE_URL happens to be sitting
# in the developer's .env (needed for actually running the app), api_server
# would silently build a real Postgres-backed graph here and every test in
# this file would be hitting a real database instead of the fast, free,
# deterministic MemorySaver fallback. An empty string (not unset) is used
# on purpose: dental_agent.py's own load_dotenv() call only fills in a
# variable that's completely absent from the environment, so an explicit
# empty string here survives that and still reads as "not configured" to
# _build_checkpointer()'s `if database_url:` check.
os.environ["DATABASE_URL"] = ""

import pytest
from fastapi.testclient import TestClient

import dental_agent as agent
import api_server
from test_dental_agent import _FakeRouterLLM, _FakeSpecialistLLM


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Same reset as test_dental_agent.py's reset_bookings fixture -- clean
    booking list, fresh fake calendar, no real disk writes."""
    agent.BOOKINGS.clear()
    agent._next_id_counter = 1000
    agent.calendar_client = agent.FakeCalendarClient()
    monkeypatch.setattr(agent, "_save_bookings_index", lambda: None)
    yield
    agent.BOOKINGS.clear()


@pytest.fixture
def client(monkeypatch):
    """A TestClient wired to a scripted fake router + FAQ specialist. Note
    this monkeypatches dental_agent's module-level router_llm/faq_subgraph
    AFTER api_server.graph was already built at import time -- that's fine
    because supervisor_node/faq_node look up those names fresh from
    dental_agent's module namespace on every call, not at graph-build time."""
    monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(["faq", "FINISH"]))
    monkeypatch.setattr(
        agent, "faq_subgraph",
        agent._build_specialist_subgraph([], _FakeSpecialistLLM("Hello! How can I help?")),
    )
    return TestClient(api_server.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["clinic"] == agent.CLINIC_NAME


def test_chat_endpoint_returns_a_reply(client):
    response = client.post("/chat", json={"message": "hi", "session_id": "test-session-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "test-session-1"
    assert body["reply"] == "Hello! How can I help?"


def test_chat_endpoint_keeps_separate_sessions_separate(client, monkeypatch):
    """Two different session_ids must not share conversation history --
    each is its own LangGraph thread_id."""
    monkeypatch.setattr(
        agent, "booking_subgraph",
        agent._build_specialist_subgraph([], _FakeSpecialistLLM("Booked!")),
    )
    # _FakeRouterLLM's script is stateful and shared across BOTH requests in
    # this test (it's one object assigned to agent.router_llm) -- each
    # one-turn conversation needs 2 router calls (route to booking, then
    # FINISH), so a 2-item script would work for the first request and then
    # just repeat "FINISH" forever for the second. Give it enough turns for
    # both independent conversations.
    monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(["booking", "FINISH", "booking", "FINISH"]))

    r1 = client.post("/chat", json={"message": "book me something", "session_id": "session-A"})
    r2 = client.post("/chat", json={"message": "book me something", "session_id": "session-B"})
    assert r1.json()["session_id"] == "session-A"
    assert r2.json()["session_id"] == "session-B"
    assert r1.json()["reply"] == "Booked!"
    assert r2.json()["reply"] == "Booked!"


def test_chat_endpoint_rejects_empty_message(client):
    response = client.post("/chat", json={"message": "", "session_id": "s1"})
    assert response.status_code == 422  # pydantic validation: min_length=1


def test_chat_endpoint_rejects_missing_session_id(client):
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 422


def test_chat_endpoint_never_leaks_a_raw_stack_trace(client, monkeypatch):
    """If something inside graph.invoke() blows up unexpectedly, the
    caller (a public chat widget, a WhatsApp webhook) must get a generic
    500 message, never a raw Python traceback."""
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated internal failure with sensitive details")

    monkeypatch.setattr(agent, "router_llm", type("X", (), {"invoke": staticmethod(_boom)})())
    response = client.post("/chat", json={"message": "hi", "session_id": "s1"})
    assert response.status_code == 500
    assert "sensitive details" not in response.text
    assert "RuntimeError" not in response.text
