"""
Test suite for dental_agent.py.

Run with:
    pytest test_dental_agent.py -v

Put this file in the same folder as dental_agent.py.

Three tiers, in increasing order of cost/flakiness:

  Tier 1 - Unit tests (TestBookingValidation, TestFindAppointment,
  TestGetCurrentDatetime, TestFaqSearch): pure Python logic, no LLM calls,
  no network, no cost, fully deterministic. Run these EVERY time you touch
  the code — they take well under a second.

  Tier 2 - Structural/integration tests (TestGraphStructure,
  TestSupervisorSafetyNet, TestMultiAgentFlowOffline): exercise the actual
  LangGraph wiring end-to-end, but with fake LLM objects standing in for
  Claude, so still zero cost and fully deterministic. Run these every time
  too — they're what would have caught the routing hang before it ever
  reached a terminal.

  Tier 3 - Live behavioral/quality scenarios (GOLDEN_SCENARIOS +
  test_golden_scenario): run the REAL model end-to-end against a fixed list
  of scenarios (Sunday requests, emergency deflection, identity checks,
  etc.) to catch quality regressions after a prompt or model change. These
  cost tokens, take longer, and are somewhat fuzzy since model wording
  varies — they're opt-in, not run by default.
"""

import os

# Force this OFF before importing dental_agent, so every ordinary test in
# this file falls back to the fast, free, deterministic MemorySaver instead
# of silently connecting to whatever real database happens to be sitting in
# DATABASE_URL (e.g. a developer's .env, needed for actually running the
# app). TestPostgresCheckpointer below is the one deliberate exception, and
# it opts back in explicitly via monkeypatch, scoped to just its own tests.
# An empty string (not unset) is used on purpose: dental_agent.py's own
# load_dotenv() call only fills in a variable that's completely absent from
# the environment, so an explicit empty string here survives that and still
# reads as "not configured" to _build_checkpointer()'s `if database_url:`.
os.environ["DATABASE_URL"] = ""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

import dental_agent as agent


class _NullBookingsStore:
    """No-op bookings store for offline tests -- BOOKINGS (the in-memory
    list) is the only source of truth during a test; nothing is persisted
    anywhere, so tests never touch a real file or database."""

    def load_all(self):
        return []

    def add(self, booking):
        pass

    def remove(self, booking_id):
        pass


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_bookings(monkeypatch):
    """Every test starts and ends with a clean in-memory booking list AND a
    fresh, empty FakeCalendarClient, so tests can never leak state into each
    other regardless of order or of what got "booked" on the fake calendar.

    Also swaps in a no-op bookings_store -- the offline tests should never
    touch a real file or database, and don't need to: they only care about
    BOOKINGS' in-memory behavior for the duration of each test."""
    agent.BOOKINGS.clear()
    agent._next_id_counter = 1000
    agent.calendar_client = agent.FakeCalendarClient()
    monkeypatch.setattr(agent, "bookings_store", _NullBookingsStore())
    yield
    agent.BOOKINGS.clear()


def _extract_id(confirmation_text: str) -> str:
    """Pull 'APT-1001' out of a book_appointment confirmation string."""
    return confirmation_text.split("ID ")[1].split(" ")[0]


# ---------------------------------------------------------------------------
# Tier 1: booking validation logic
# ---------------------------------------------------------------------------

class TestBookingValidation:
    def test_rejects_malformed_date(self):
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "16-09-2026", "time": "10:00"}
        )
        assert "not a valid date" in result

    def test_rejects_malformed_time(self):
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "10h00"}
        )
        assert "not a valid time" in result

    def test_rejects_sunday(self):
        # 2026-09-20 is a Sunday.
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-20", "time": "10:00"}
        )
        assert "closed on Sundays" in result

    def test_rejects_saturday_after_hours(self):
        # 2026-09-19 is a Saturday; clinic closes at 14:00 that day.
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-19", "time": "15:00"}
        )
        assert "outside working hours" in result

    def test_accepts_saturday_within_hours(self):
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-19", "time": "09:30"}
        )
        assert "confirmed" in result.lower()

    def test_rejects_weekday_evening(self):
        # Weekday closing time is 19:00.
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "20:00"}
        )
        assert "outside working hours" in result

    def test_rejects_double_booking(self):
        agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "09:00"}
        )
        result = agent.book_appointment.invoke(
            {"patient_name": "B", "phone": "2", "date": "2026-09-16", "time": "09:00"}
        )
        assert "already booked" in result

    def test_cancel_frees_the_slot(self):
        confirmation = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-17", "time": "10:30"}
        )
        appt_id = _extract_id(confirmation)
        before = agent.get_available_slots.invoke({"date": "2026-09-17"})
        assert "10:30" not in before

        cancel_result = agent.cancel_appointment.invoke({"appointment_id": appt_id})
        assert "cancelled" in cancel_result

        after = agent.get_available_slots.invoke({"date": "2026-09-17"})
        assert "10:30" in after

    def test_cancel_unknown_id(self):
        result = agent.cancel_appointment.invoke({"appointment_id": "APT-9999"})
        assert "No appointment found" in result


# ---------------------------------------------------------------------------
# Tier 1: find_appointment identity-verification logic
# ---------------------------------------------------------------------------

class TestFindAppointment:
    def _seed(self):
        agent.book_appointment.invoke(
            {"patient_name": "Maria Garcia", "phone": "600111222",
             "date": "2026-09-16", "time": "09:00", "service": "Dental cleaning"}
        )
        agent.book_appointment.invoke(
            {"patient_name": "Maria Garcia", "phone": "600333444",
             "date": "2026-09-17", "time": "10:30", "service": "Filling"}
        )

    def test_lookup_by_id_alone_is_sufficient(self):
        confirmation = agent.book_appointment.invoke(
            {"patient_name": "John Smith", "phone": "600555666",
             "date": "2026-09-18", "time": "09:00"}
        )
        appt_id = _extract_id(confirmation)
        result = agent.find_appointment.invoke({"appointment_id": appt_id})
        assert "John Smith" in result

    def test_single_criterion_is_rejected(self):
        self._seed()
        result = agent.find_appointment.invoke({"patient_name": "Maria Garcia"})
        assert "at least two" in result
        assert "Maria Garcia" not in result  # must not leak booking details

    def test_date_without_time_does_not_count_as_a_criterion(self):
        self._seed()
        result = agent.find_appointment.invoke({"date": "2026-09-16"})
        assert "at least two" in result

    def test_two_criteria_disambiguates_same_name(self):
        self._seed()
        result = agent.find_appointment.invoke(
            {"patient_name": "Maria Garcia", "phone": "600333444"}
        )
        assert "2026-09-17" in result
        assert "2026-09-16" not in result

    def test_no_match_found(self):
        self._seed()
        result = agent.find_appointment.invoke(
            {"patient_name": "Nobody Here", "phone": "000000000"}
        )
        assert "No appointment found" in result


# ---------------------------------------------------------------------------
# Tier 1: clock and FAQ tools
# ---------------------------------------------------------------------------

class TestGetCurrentDatetime:
    def test_returns_a_barcelona_timestamp(self):
        from datetime import datetime
        result = agent.get_current_datetime.invoke({})
        assert "Barcelona" in result
        # Sanity-check it actually contains today's real date, not a stale
        # hardcoded value.
        today_str = datetime.now(agent.CLINIC_TIMEZONE).strftime("%Y-%m-%d")
        assert today_str in result


class TestFaqSearch:
    @pytest.mark.parametrize("query,expected_keyword", [
        ("how much does a cleaning cost", "checkup"),
        ("is there parking nearby", "parking"),
        ("do you take Sanitas", "Adeslas"),  # insurance answer lists several accepted insurers
        ("where is the clinic located", "Barcelona"),
    ])
    def test_known_topics_match(self, query, expected_keyword):
        result = agent.search_clinic_faq.invoke({"query": query})
        assert expected_keyword.lower() in result.lower()

    def test_unmatched_query_gives_fallback(self):
        result = agent.search_clinic_faq.invoke({"query": "do you sell toothbrushes"})
        assert "Nothing in the clinic FAQ" in result
        assert agent.CLINIC_PHONE in result


# ---------------------------------------------------------------------------
# Tier 1: bookings-index persistence (regression tests for the "restart
# wipes the agent's memory of existing bookings" bug)
# ---------------------------------------------------------------------------

class TestBookingsIndexPersistence:
    """Tests JsonFileBookingsStore -- the fallback backend used when
    DATABASE_URL isn't set. Each test builds its own store pointed at a
    pytest tmp_path, so none of this touches a real file. See
    TestPostgresBookingsStore below for the real production backend."""

    def test_save_then_load_roundtrip(self, tmp_path):
        index_file = tmp_path / "bookings_index.json"
        store = agent.JsonFileBookingsStore(index_file)

        agent.BOOKINGS[:] = [
            {"id": "APT-1001", "event_id": "evt-1", "patient_name": "Dave Mai",
             "phone": "666666666", "date": "2026-09-17", "time": "18:00", "service": "Orthodontics consultation"},
        ]
        store.add(agent.BOOKINGS[0])

        assert index_file.exists()
        loaded = store.load_all()
        assert loaded == agent.BOOKINGS

    def test_find_appointment_survives_a_simulated_restart(self, tmp_path, monkeypatch):
        """The exact bug: book an appointment, simulate the process
        restarting (BOOKINGS reset to whatever the store's load_all()
        returns), then confirm find_appointment can still see it."""
        index_file = tmp_path / "bookings_index.json"
        store = agent.JsonFileBookingsStore(index_file)
        monkeypatch.setattr(agent, "bookings_store", store)

        confirmation = agent.book_appointment.invoke(
            {"patient_name": "Dave Mai", "phone": "666666666",
             "date": "2026-09-17", "time": "18:00", "service": "Orthodontics consultation"}
        )
        appt_id = _extract_id(confirmation)

        # Simulate a fresh process: nothing left in memory except what
        # gets reloaded from disk.
        agent.BOOKINGS[:] = store.load_all()

        result = agent.find_appointment.invoke({"patient_name": "Dave Mai", "phone": "666666666"})
        assert appt_id in result
        assert "Dave Mai" in result

    def test_next_counter_start_resumes_after_highest_loaded_id(self):
        loaded = [{"id": "APT-1003"}, {"id": "APT-1007"}, {"id": "APT-1002"}]
        assert agent._next_counter_start(loaded) == 1007

    def test_next_counter_start_defaults_to_1000_when_empty(self):
        assert agent._next_counter_start([]) == 1000

    def test_corrupted_index_file_falls_back_to_empty_list(self, tmp_path):
        index_file = tmp_path / "bookings_index.json"
        index_file.write_text("{not valid json", encoding="utf-8")
        store = agent.JsonFileBookingsStore(index_file)
        assert store.load_all() == []


@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to a real (throwaway) Postgres database to run these",
)
class TestPostgresBookingsStore:
    """Opt-in integration tests against a REAL Postgres database (never run
    by default). Verifies the exact thing Stage 2 of the Neon migration is
    for: that the booking index survives a brand new process pointed at
    the same database -- the original bug report ("booked an appointment,
    closed the agent, reopened it, and it couldn't find my booking") --
    now with the index itself in Postgres instead of a local JSON file.

    Run with, e.g., the same local Postgres used for TestPostgresCheckpointer:
        TEST_DATABASE_URL=postgresql://postgres:testpass@localhost:5432/dental_agent_test \\
            pytest test_dental_agent.py -v -k PostgresBookingsStore

    Never point this at your real Neon database -- it creates and reads
    real booking rows. Use a disposable local/test database instead.
    """

    def setup_method(self):
        os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
        # Start every test with an empty `bookings` table (creating it via
        # PostgresBookingsStore's constructor if it doesn't exist yet), so
        # these tests never see leftover rows from a previous run.
        pool = agent._get_pg_pool()
        with pool.connection() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS bookings ("
                "id TEXT PRIMARY KEY, event_id TEXT NOT NULL, patient_name TEXT NOT NULL, "
                "phone TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL, service TEXT NOT NULL)"
            )
            conn.execute("DELETE FROM bookings")

    def teardown_method(self):
        agent.close_database()
        os.environ.pop("DATABASE_URL", None)

    def test_build_bookings_store_uses_postgres_when_database_url_is_set(self):
        store = agent._build_bookings_store()
        assert isinstance(store, agent.PostgresBookingsStore)

    def test_add_load_remove_roundtrip(self):
        store = agent.PostgresBookingsStore(agent._get_pg_pool())
        booking = {
            "id": "APT-9001", "event_id": "evt-9001", "patient_name": "Dave Mai",
            "phone": "666666666", "date": "2026-09-17", "time": "18:00",
            "service": "Orthodontics consultation",
        }
        store.add(booking)
        assert store.load_all() == [booking]

        store.remove("APT-9001")
        assert store.load_all() == []

    def test_table_creation_is_idempotent(self):
        agent.PostgresBookingsStore(agent._get_pg_pool())
        agent.PostgresBookingsStore(agent._get_pg_pool())  # must not raise

    def test_find_appointment_survives_a_simulated_restart(self, monkeypatch):
        """The original bug report, end to end: book an appointment with
        the booking index in Postgres, close the process (close_database),
        start a brand new one pointed at the same database, and confirm
        find_appointment can still see the booking."""
        monkeypatch.setattr(agent, "bookings_store", agent._build_bookings_store())
        agent.BOOKINGS[:] = agent.bookings_store.load_all()

        confirmation = agent.book_appointment.invoke(
            {"patient_name": "Dave Mai", "phone": "666666666",
             "date": "2026-09-17", "time": "18:00", "service": "Orthodontics consultation"}
        )
        appt_id = _extract_id(confirmation)

        # Simulate the process exiting and a brand new one starting up,
        # pointed at the same database.
        agent.close_database()
        fresh_store = agent._build_bookings_store()
        agent.BOOKINGS[:] = fresh_store.load_all()

        result = agent.find_appointment.invoke({"patient_name": "Dave Mai", "phone": "666666666"})
        assert appt_id in result
        assert "Dave Mai" in result


# ---------------------------------------------------------------------------
# Tier 1: calendar-backed booking tools
# ---------------------------------------------------------------------------

class _BrokenCalendarClient:
    """Simulates the calendar being unreachable (network error, auth
    failure, quota exceeded, ...) -- every method raises."""

    def get_busy_intervals(self, day):
        raise RuntimeError("simulated calendar outage")

    def create_event(self, **kwargs):
        raise RuntimeError("simulated calendar outage")

    def delete_event(self, event_id):
        raise RuntimeError("simulated calendar outage")


class TestCalendarBackedTools:
    def test_get_available_slots_reports_closed_day_directly(self):
        # 2026-09-20 is a Sunday.
        result = agent.get_available_slots.invoke({"date": "2026-09-20"})
        assert "closed" in result.lower()

    def test_booking_creates_a_real_calendar_event(self):
        confirmation = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "09:00"}
        )
        appt_id = _extract_id(confirmation)
        booking = agent._find_booking(appt_id)
        assert booking["event_id"] in agent.calendar_client.events
        event = agent.calendar_client.events[booking["event_id"]]
        assert "A" in event["summary"]

    def test_cancel_removes_the_calendar_event(self):
        confirmation = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "09:00"}
        )
        appt_id = _extract_id(confirmation)
        event_id = agent._find_booking(appt_id)["event_id"]
        agent.cancel_appointment.invoke({"appointment_id": appt_id})
        assert event_id not in agent.calendar_client.events

    def test_calendar_outage_on_availability_check_is_reported_gracefully(self, monkeypatch):
        monkeypatch.setattr(agent, "calendar_client", _BrokenCalendarClient())
        result = agent.get_available_slots.invoke({"date": "2026-09-16"})
        assert "trouble reaching" in result.lower()
        assert agent.CLINIC_PHONE in result

    def test_calendar_outage_on_booking_does_not_silently_succeed(self, monkeypatch):
        """If the calendar is down, the tool must NOT create a local booking
        record and claim success -- that would be a phantom appointment the
        clinic can't see anywhere."""
        monkeypatch.setattr(agent, "calendar_client", _BrokenCalendarClient())
        result = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "09:00"}
        )
        assert "confirmed" not in result.lower()
        assert agent.CLINIC_PHONE in result
        assert agent.BOOKINGS == []

    def test_calendar_outage_on_cancel_keeps_the_local_record(self, monkeypatch):
        """If the calendar delete fails, the local booking must NOT be
        removed -- otherwise the agent would forget an appointment that's
        still sitting on the real calendar."""
        confirmation = agent.book_appointment.invoke(
            {"patient_name": "A", "phone": "1", "date": "2026-09-16", "time": "09:00"}
        )
        appt_id = _extract_id(confirmation)
        monkeypatch.setattr(agent, "calendar_client", _BrokenCalendarClient())
        result = agent.cancel_appointment.invoke({"appointment_id": appt_id})
        assert "couldn't cancel" in result.lower()
        assert agent._find_booking(appt_id) is not None


# ---------------------------------------------------------------------------
# Tier 1: extract_text / last_reply_text content-normalization helpers
# ---------------------------------------------------------------------------

class TestLastReplyText:
    def test_plain_string_content(self):
        messages = [HumanMessage(content="hi"), AIMessage(content="Hello!")]
        assert agent.last_reply_text(messages) == "Hello!"

    def test_skips_trailing_empty_ai_message(self):
        """The exact real-world shape that caused the original bug: a real
        answer, then a later AIMessage with empty list content and no text."""
        messages = [
            HumanMessage(content="Can I get an appointment this Sunday?"),
            AIMessage(content=[{"type": "tool_use", "id": "t1", "name": "get_current_datetime", "input": {}}]),
            ToolMessage(content="Wednesday, 2026-09-16", tool_call_id="t1"),
            AIMessage(content="We're closed Sundays -- how about Saturday or Monday instead?"),
            AIMessage(content=[]),  # genuinely empty completion from a needless extra hop
        ]
        assert agent.last_reply_text(messages) == (
            "We're closed Sundays -- how about Saturday or Monday instead?"
        )

    def test_all_empty_falls_back_to_generic_message(self):
        messages = [HumanMessage(content="hi"), AIMessage(content=""), AIMessage(content=[])]
        result = agent.last_reply_text(messages)
        assert result  # never blank
        assert "repeat" in result.lower()


# ---------------------------------------------------------------------------
# Tier 2: graph structure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    def test_build_graph_does_not_raise(self):
        agent.build_graph()

    def test_top_level_routing_edges(self):
        graph = agent.build_graph()
        g = graph.get_graph()
        edge_pairs = {(e.source, e.target) for e in g.edges}
        assert ("__start__", "supervisor") in edge_pairs
        assert ("supervisor", "booking") in edge_pairs
        assert ("supervisor", "faq") in edge_pairs
        assert ("supervisor", "__end__") in edge_pairs
        # Specialists must return control to the supervisor.
        assert ("booking", "supervisor") in edge_pairs
        assert ("faq", "supervisor") in edge_pairs

    def test_specialist_subgraphs_have_agent_and_tools_nodes(self):
        for subgraph in (agent.booking_subgraph, agent.faq_subgraph):
            nodes = set(subgraph.get_graph().nodes.keys())
            assert "agent" in nodes
            assert "tools" in nodes


# ---------------------------------------------------------------------------
# Tier 2: conversation-memory checkpointer (MemorySaver <-> Postgres)
# ---------------------------------------------------------------------------

class TestCheckpointerFallback:
    """Without DATABASE_URL set, everything must keep working exactly like
    before the Postgres migration -- zero setup, zero network, a fresh
    in-memory checkpointer every time. This is what every other test in
    this file (and test_api_server.py) relies on running offline."""

    def test_build_graph_falls_back_to_memory_saver_without_database_url(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        graph = agent.build_graph()
        assert isinstance(graph.checkpointer, MemorySaver)

    def test_close_checkpointer_is_a_harmless_noop_on_memory_saver(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        agent.build_graph()
        agent.close_database()  # must not raise


@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to a real (throwaway) Postgres database to run these",
)
class TestPostgresCheckpointer:
    """Opt-in integration tests against a REAL Postgres database (never run
    by default -- these need an actual database, unlike everything else in
    this file). Verifies the exact thing the Neon migration is for: that
    conversation history survives a brand new process pointed at the same
    database, which is what happens on every Render redeploy/restart.

    Run with, e.g., a local Postgres:
        TEST_DATABASE_URL=postgresql://postgres:testpass@localhost:5432/dental_agent_test \\
            pytest test_dental_agent.py -v -k Postgres

    Never point this at your real Neon database -- it creates and reads
    real checkpoint rows. Use a disposable local/test database instead.
    """

    def setup_method(self):
        os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]

    def teardown_method(self):
        agent.close_database()
        os.environ.pop("DATABASE_URL", None)

    def test_build_graph_uses_postgres_saver_when_database_url_is_set(self):
        graph = agent.build_graph()
        assert isinstance(graph.checkpointer, PostgresSaver)

    def test_setup_is_idempotent(self):
        # Calling build_graph() twice re-runs checkpointer.setup() against
        # tables that already exist -- must not raise.
        agent.build_graph()
        agent.close_database()
        agent.build_graph()

    def test_conversation_history_survives_a_simulated_restart(self, monkeypatch):
        """The regression this whole migration is for: book a conversation's
        history under one 'process' (one build_graph() + ExitStack), close
        it exactly like a real process exiting would, then start a brand
        new 'process' pointed at the same database and confirm the history
        is still there -- unlike MemorySaver, which would have lost it."""
        monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(["faq", "FINISH"]))
        monkeypatch.setattr(
            agent, "faq_subgraph",
            agent._build_specialist_subgraph([], _FakeSpecialistLLM("Here's the info.")),
        )
        thread_id = "test-restart-thread"
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}

        graph_1 = agent.build_graph()
        graph_1.invoke(
            {"messages": [HumanMessage(content="what are your hours?")], "specialist_hops": 0},
            config=config,
        )
        agent.close_database()  # simulates the process exiting

        # Brand new checkpointer/connection, same database -- like a fresh
        # `python dental_agent.py` or a Render redeploy.
        graph_2 = agent.build_graph()
        state = graph_2.get_state(config)
        contents = [m.content for m in state.values["messages"]]
        assert "what are your hours?" in contents
        assert "Here's the info." in contents

    def test_recovers_from_a_dropped_connection(self, monkeypatch):
        """Regression test for the real bug this class exists to catch: a
        single long-held Postgres connection eventually goes stale (Neon
        suspending its compute when idle, a network blip, Render's server
        sleeping/waking) and every request after that fails with
        'the connection is closed' until the process is restarted. Using a
        connection pool (see _build_checkpointer()) fixes this by
        validating and swapping out a dead connection the moment it's
        checked out, so this test kills the live connection out from under
        the app and confirms the very next call still succeeds."""
        # Two separate graph.invoke() calls below -> two full supervisor
        # turns -> the stateful fake router needs enough scripted steps for
        # BOTH turns (route to faq, then FINISH -- twice), same reasoning
        # as test_api_server.py's multi-session test.
        monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(["faq", "FINISH", "faq", "FINISH"]))
        monkeypatch.setattr(
            agent, "faq_subgraph",
            agent._build_specialist_subgraph([], _FakeSpecialistLLM("Here's the info.")),
        )
        config = {"configurable": {"thread_id": "test-dropped-connection"}, "recursion_limit": 12}

        graph = agent.build_graph()
        graph.invoke(
            {"messages": [HumanMessage(content="hi")], "specialist_hops": 0},
            config=config,
        )

        # Simulate the connection dying underneath the app -- kill every
        # connection actually sitting in the pool right now.
        pool = agent._pg_pool
        assert isinstance(pool, ConnectionPool)
        with pool.connection():
            pass  # make sure at least one connection has been opened
        for conn in list(pool._pool):
            conn.close()

        # Must NOT raise -- the pool should hand back a fresh, healthy
        # connection instead of the dead one.
        result = graph.invoke(
            {"messages": [HumanMessage(content="hi again")], "specialist_hops": 0},
            config=config,
        )
        assert result["messages"][-1].content == "Here's the info."


# ---------------------------------------------------------------------------
# Tier 2: the hop-cap safety net (regression test for the hang we fixed)
# ---------------------------------------------------------------------------

class TestNoBulkBookingAccess:
    """Structural guardrail: there must be no tool, and no code path, that
    can return every patient's bookings at once. This is what actually
    backs the 'unauthorized_data_access_list_all' scenario in
    eval_scenarios.py -- the agent has no way to comply even if a clever
    prompt convinced it to, because the capability doesn't exist."""

    def test_no_tool_name_suggests_bulk_access(self):
        all_tool_names = {t.name for t in agent.BOOKING_TOOLS} | {t.name for t in agent.FAQ_TOOLS}
        forbidden_terms = ["list_all", "get_all", "all_appointments", "all_bookings", "dump", "export"]
        for name in all_tool_names:
            for term in forbidden_terms:
                assert term not in name.lower(), f"Tool '{name}' looks like it could expose bulk booking data"

    def test_find_appointment_with_no_arguments_reveals_nothing(self):
        agent.book_appointment.invoke(
            {"patient_name": "Secret Patient", "phone": "999999999",
             "date": "2026-09-16", "time": "09:00"}
        )
        result = agent.find_appointment.invoke({})
        assert "at least two" in result
        assert "Secret Patient" not in result


class TestSupervisorSafetyNet:
    def test_at_cap_forces_finish_without_calling_the_llm(self, monkeypatch):
        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("router_llm should not be invoked once the hop cap is reached")
        monkeypatch.setattr(agent, "router_llm", type("X", (), {"invoke": staticmethod(_should_not_be_called)})())

        state = {
            "messages": [HumanMessage(content="test")],
            "specialist_hops": agent.MAX_SPECIALIST_HOPS_PER_TURN,
        }
        result = agent.supervisor_node(state)
        assert result == {"next": "FINISH"}


class _FakeRoute:
    def __init__(self, next):
        self.next = next


class _FakeRouterLLM:
    """Returns a scripted sequence of routing decisions, repeating the last
    one forever once the script runs out (simulates a router that keeps
    wanting to route rather than finish).

    Matches the real router_llm's contract: with_structured_output(...,
    include_raw=True) returns a dict with "parsed" (the RouteDecision-like
    object) and "raw" (the underlying AIMessage, used for token-usage
    accounting). A plain AIMessage with no usage_metadata set is fine here
    -- _usage_record() just skips it, so offline tests report zero cost."""
    def __init__(self, sequence):
        self._seq = list(sequence)
        self._i = 0

    def invoke(self, messages):
        val = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return {"parsed": _FakeRoute(val), "raw": AIMessage(content="")}


class _FakeSpecialistLLM:
    def __init__(self, reply_text="OK, done.", on_call=None):
        self.reply_text = reply_text
        self._on_call = on_call

    def invoke(self, messages):
        if self._on_call:
            self._on_call()
        return AIMessage(content=self.reply_text)


class TestMultiAgentFlowOffline:
    """Full end-to-end graph runs with fake LLMs standing in for Claude:
    zero API cost, fully deterministic, and exercises the real routing,
    message-passing, and hop-counting logic."""

    def _run(self, monkeypatch, router_sequence, booking_reply="Booked!", faq_reply="Here's the info."):
        monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(router_sequence))
        monkeypatch.setattr(
            agent, "booking_subgraph",
            agent._build_specialist_subgraph([], _FakeSpecialistLLM(booking_reply)),
        )
        monkeypatch.setattr(
            agent, "faq_subgraph",
            agent._build_specialist_subgraph([], _FakeSpecialistLLM(faq_reply)),
        )
        graph = agent.build_graph()
        config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 12}
        return graph.invoke(
            {"messages": [HumanMessage(content="hello")], "specialist_hops": 0},
            config=config,
        )

    def test_single_hop_to_booking_then_finish(self, monkeypatch):
        result = self._run(monkeypatch, router_sequence=["booking", "FINISH"])
        assert result["messages"][-1].content == "Booked!"
        assert result["specialist_hops"] == 1

    def test_mixed_intent_visits_both_specialists(self, monkeypatch):
        result = self._run(
            monkeypatch,
            router_sequence=["booking", "faq", "FINISH"],
            booking_reply="Booked your slot.",
            faq_reply="And yes, we take Sanitas.",
        )
        assert result["specialist_hops"] == 2
        # Both replies should have made it into the shared conversation.
        contents = [m.content for m in result["messages"]]
        assert "Booked your slot." in contents
        assert "And yes, we take Sanitas." in contents

    def test_second_specialist_empty_completion_falls_back_to_first_reply(self, monkeypatch):
        """Regression test for the 'blank final reply' bug: the supervisor
        can take a needless second hop after a single-intent message is
        already fully answered, and the model can legitimately return a
        genuinely empty completion when asked to react to a conversation
        that doesn't need anything more from it. When that happens, the
        patient must still see the earlier, real answer -- not silence."""
        result = self._run(
            monkeypatch,
            router_sequence=["booking", "faq", "FINISH"],
            booking_reply="We're closed Sundays, how about Saturday instead?",
            faq_reply="",  # simulates a real empty AIMessage completion
        )
        assert agent.last_reply_text(result["messages"]) == (
            "We're closed Sundays, how about Saturday instead?"
        )

    def test_runaway_router_is_capped(self, monkeypatch):
        """Regression test for the hang: a router that ALWAYS wants to route
        to booking and never says FINISH must still be stopped at
        MAX_SPECIALIST_HOPS_PER_TURN specialist calls, not 12+ (recursion
        limit) or worse."""
        call_count = {"n": 0}

        def _count():
            call_count["n"] += 1

        monkeypatch.setattr(agent, "router_llm", _FakeRouterLLM(["booking"]))  # never says FINISH
        monkeypatch.setattr(
            agent, "booking_subgraph",
            agent._build_specialist_subgraph([], _FakeSpecialistLLM("still working...", on_call=_count)),
        )
        graph = agent.build_graph()
        config = {"configurable": {"thread_id": "test-runaway"}, "recursion_limit": 12}
        result = graph.invoke(
            {"messages": [HumanMessage(content="anything")], "specialist_hops": 0},
            config=config,
        )
        assert call_count["n"] <= agent.MAX_SPECIALIST_HOPS_PER_TURN
        assert result["specialist_hops"] == agent.MAX_SPECIALIST_HOPS_PER_TURN


# ---------------------------------------------------------------------------
# Tier 3: live behavioral/quality scenarios (real API, costs tokens, opt-in)
# ---------------------------------------------------------------------------
#
# Run explicitly with:
#     RUN_LIVE_TESTS=1 pytest test_dental_agent.py -v -k golden_scenario
#
# Do this after changing a system prompt, switching models, or before
# showing this to someone else -- not on every save. Assertions here are
# deliberately loose (keyword/structure checks), since exact wording from
# the model will vary between runs.

LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"

# Scenario definitions live in eval_scenarios.py, shared with run_evals.py
# (which produces the scored report with cost/latency). Keeping one source
# of truth means a new scenario you add is automatically covered by both
# the quick pytest pass/fail check here and the detailed report there.
from eval_scenarios import ALL_SCENARIOS  # noqa: E402


@pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_TESTS=1 (and a real ANTHROPIC_API_KEY) to run live scenarios")
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s["name"] for s in ALL_SCENARIOS])
def test_golden_scenario(scenario):
    graph = agent.build_graph()
    config = {"configurable": {"thread_id": f"golden-{scenario['name']}"}, "recursion_limit": 12}

    reply = ""
    for turn in scenario["turns"]:
        result = graph.invoke(
            {"messages": [HumanMessage(content=turn)], "specialist_hops": 0},
            config=config,
        )
        reply = agent.last_reply_text(result["messages"])

    reply_lower = reply.lower()
    required = scenario["must_contain_any"]
    if required:  # an empty list means "no positive requirement", not "must match nothing"
        assert any(kw.lower() in reply_lower for kw in required), (
            f"Scenario '{scenario['name']}': expected one of {required} "
            f"in the reply, got:\n{reply}"
        )
    for forbidden in scenario["must_not_contain"]:
        assert forbidden.lower() not in reply_lower, (
            f"Scenario '{scenario['name']}': forbidden text '{forbidden}' found in reply:\n{reply}"
        )
