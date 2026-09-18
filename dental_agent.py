"""
Dental Clinic Multi-Agent Assistant
=====================================

A multi-agent LangGraph system for a dental clinic receptionist ("Laia") in
Barcelona. Three agents, each built with LangGraph's StateGraph:

    1. Supervisor  — classifies the patient's intent and routes to a
                      specialist. Never talks to the patient directly.
    2. Booking     — its own ReAct sub-agent (agent/tools loop) that checks
                      availability, books, finds, cancels, and reschedules
                      appointments.
    3. FAQ         — its own ReAct sub-agent that answers general clinic
                      questions (services, prices, location, parking,
                      insurance) from a small hardcoded knowledge base, and
                      handles greetings/small talk.

Top-level graph shape:

              +---------------------+
              |                     |
              v                     |
    START -> supervisor ----------- +
              |    |
        booking    faq
         (sub)    (sub)
              |    |
              +----+---> (edge back to supervisor)
              |
        supervisor decides: route again, or "FINISH" -> END

Each specialist is itself a compiled StateGraph following the standard
LangGraph ReAct pattern (agent node -> tools_condition -> tools node -> back
to agent node -> ... -> END), invoked fresh on the full conversation each
time the supervisor hands off to it, with its own system prompt and its own
tools. After it produces a final answer, control returns to the supervisor,
which decides whether another specialist is still needed this turn or
whether to finish and let the patient see the reply.

Run it:
    pip install -r requirements.txt
    Set ANTHROPIC_API_KEY (env var or .env file)
    python dental_agent.py

Then chat with it interactively. Type "exit" or "quit" to stop.
"""

import json
import operator
import os
import sys
import time
import uuid
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError

from calendar_client import CalendarClient, GoogleCalendarClient, FakeCalendarClient

load_dotenv()



# Block types we know carry no user-facing text, and are safe to silently
# skip when normalizing content. Anything NOT in this list and NOT a
# recognized text block is "unrecognized" -- see extract_text below for why
# we deliberately do NOT silently drop those too.
_KNOWN_NON_TEXT_BLOCK_TYPES = {
    "thinking", "redacted_thinking", "tool_use", "tool_result",
    "server_tool_use", "web_search_tool_result", "code_execution_tool_result",
}


def extract_text(message) -> str:
    """Normalize an AIMessage's .content into a plain string.

    A real Anthropic response's content is NOT always a plain string --
    langchain_anthropic sometimes represents it as a list of content blocks
    (e.g. a text block alongside other block types the API returned), and
    that shape isn't fully predictable per-response. Code that assumes
    `.content` is always `str` (an f-string print, a `.lower()` call in a
    test) will work most of the time and then crash without warning on
    whichever response happens to come back as a list. Always go through
    this helper instead of touching `.content` directly.

    Deliberately fail LOUD, not silent: if a block's shape isn't one we
    recognize (not a plain string, not a {"type": "text", "text": ...}
    dict, not a known non-text block type), we do NOT quietly drop it and
    return "". A silently-empty reply is much worse than an ugly one --
    it masks real behavior in tests/evals and shows the patient nothing.
    Instead we surface the raw block(s) so the shape is visible immediately
    (in a test failure, an eval report, or the transcript) instead of
    requiring a separate debugging round-trip.
    """
    content = message.content if hasattr(message, "content") else message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        unrecognized = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            # Handle both plain dicts (the common langchain_anthropic shape)
            # and any object exposing .type/.text attributes instead, just
            # in case a future/alternate SDK path returns typed objects.
            if isinstance(block, dict):
                block_type = block.get("type")
                text_val = block.get("text")
            else:
                block_type = getattr(block, "type", None)
                text_val = getattr(block, "text", None)

            if block_type == "text" and text_val:
                parts.append(text_val)
            elif block_type in _KNOWN_NON_TEXT_BLOCK_TYPES:
                continue
            else:
                unrecognized.append(block)

        if parts:
            return "".join(parts)
        if unrecognized:
            return f"[extract_text: no text block found -- raw content: {unrecognized!r}]"
        return ""
    return str(content)


def last_reply_text(messages: list) -> str:
    """Find the text the patient should actually see at the end of a turn.

    `result["messages"][-1]` is NOT always a usable reply. A specialist can
    legitimately produce a genuinely empty completion (content == "" or
    == []) when the supervisor routes it to "react" to a conversation that
    doesn't actually need anything more from it -- e.g. it gets invoked
    again right after another specialist (or itself) already gave a
    complete answer, with no new human input since, and the model decides
    there's nothing to add. When that happens, the good answer is still
    sitting a couple of messages back; showing the patient nothing instead
    of it is a real bug, not just a cosmetic one.

    This walks backward from the end of the conversation and returns the
    text of the first AIMessage whose extracted text is non-blank. Only as
    a last resort (no AIMessage anywhere has any text) does it fall back to
    a generic apology, so a genuinely broken turn still shows *something*
    sane to the patient instead of silence.
    """
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = extract_text(msg)
            if text.strip():
                return text
    return (
        "Sorry, I didn't quite catch that -- could you repeat your last "
        "message?"
    )


# ---------------------------------------------------------------------------
# Clinic configuration
# ---------------------------------------------------------------------------

CLINIC_NAME = "Bright Smile Dental Clinic"
CLINIC_PHONE = "+34 93 000 00 00"  # TODO: replace with the real clinic phone number
CLINIC_TIMEZONE = ZoneInfo("Europe/Madrid")  # Barcelona's timezone (handles CET/CEST automatically)

SERVICES = [
    "Dental cleaning",
    "General checkup",
    "Filling",
    "Extraction",
    "Orthodontics consultation",
]

# weekday() -> (opening time, closing time). Monday=0 ... Sunday=6.
# A day missing from this dict is treated as closed (e.g. Sunday).
WORKING_HOURS: dict[int, tuple[dt_time, dt_time]] = {
    0: (dt_time(9, 0), dt_time(19, 0)),  # Monday
    1: (dt_time(9, 0), dt_time(19, 0)),  # Tuesday
    2: (dt_time(9, 0), dt_time(19, 0)),  # Wednesday
    3: (dt_time(9, 0), dt_time(19, 0)),  # Thursday
    4: (dt_time(9, 0), dt_time(19, 0)),  # Friday
    5: (dt_time(9, 0), dt_time(14, 0)),  # Saturday
    # 6 (Sunday) intentionally omitted: clinic is closed.
}

# TODO: replace every value below with the clinic's real information — this
# is placeholder/example content only, clearly not real prices or an address.
FAQ_KB: list[dict] = [
    {
        "topic": "services",
        "keywords": ["service", "services", "treatment", "treatments", "offer",
                     "cleaning", "checkup", "check-up", "filling", "extraction",
                     "orthodontics", "ortho", "braces"],
        "answer": f"We offer: {', '.join(SERVICES)}.",
    },
    {
        "topic": "prices",
        "keywords": ["price", "prices", "cost", "costs", "fee", "fees",
                     "how much", "euro", "€", "expensive", "cheap"],
        "answer": (
            "[EXAMPLE PRICES — replace with real ones] General checkup: €40. "
            "Dental cleaning: €60. Filling: from €70. Extraction: from €90. "
            "Orthodontics consultation: €50 (often deducted from treatment "
            "cost if the patient proceeds). Final cost is always confirmed "
            "at the visit."
        ),
    },
    {
        "topic": "location",
        "keywords": ["location", "address", "where", "located", "directions",
                      "map"],
        "answer": (
            f"[EXAMPLE ADDRESS — replace with the real one] {CLINIC_NAME} is "
            "located at Carrer de Balmes 123, 08008 Barcelona, close to "
            "Diagonal metro station (L3/L5)."
        ),
    },
    {
        "topic": "parking",
        "keywords": ["parking", "park", "car park", "garage"],
        "answer": (
            "[EXAMPLE — replace with real info] There is no dedicated clinic "
            "parking, but a public car park is a short walk away, and there "
            "is metered street parking nearby on weekdays."
        ),
    },
    {
        "topic": "insurance",
        "keywords": ["insurance", "insurer", "cover", "covered", "adeslas",
                     "sanitas", "dkv", "asisa", "mapfre"],
        "answer": (
            "[EXAMPLE — replace with the clinic's real accepted insurers] We "
            "accept most major Spanish dental insurers, including Adeslas, "
            "Sanitas, DKV, Asisa, and Mapfre. Coverage and co-pays depend on "
            "the patient's specific policy, so ask them to bring their "
            "insurance card to the appointment to confirm exact coverage."
        ),
    },
]

# ---------------------------------------------------------------------------
# Booking backend — a real calendar is the source of truth for availability
# and bookings; the local list below is just a fast index (by our own
# human-friendly appointment IDs) on top of it. See calendar_client.py for
# the CalendarClient interface, the real Google Calendar implementation, and
# setup instructions.
# ---------------------------------------------------------------------------

# How long a single appointment slot is. Used both to generate candidate
# slot start times for get_available_slots and as the event duration when
# booking. Change this if the clinic wants a different default length.
SLOT_MINUTES = 30


def _build_calendar_client() -> CalendarClient:
    """Build the real Google Calendar client if it's configured, otherwise
    fall back to an in-memory stand-in (with a loud warning) so the script
    and the offline tests still run without any Google setup. See
    calendar_client.py's module docstring for how to configure the real one."""
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    calendar_id = os.environ.get("GOOGLE_CALENDAR_ID")
    if service_account_file and calendar_id:
        return GoogleCalendarClient(calendar_id, service_account_file, CLINIC_TIMEZONE)
    print(
        "WARNING: GOOGLE_SERVICE_ACCOUNT_FILE / GOOGLE_CALENDAR_ID not set -- "
        "using an in-memory calendar stand-in instead of real Google Calendar. "
        "Bookings will NOT persist between runs until this is configured. "
        "See calendar_client.py for setup instructions.",
        file=sys.stderr,
    )
    return FakeCalendarClient()


# Module-level so tests can monkeypatch it (same pattern as router_llm /
# booking_subgraph / faq_subgraph below) -- swap in a FakeCalendarClient()
# per test for a clean, deterministic, network-free calendar every time.
calendar_client: CalendarClient = _build_calendar_client()

# Every confirmed booking, as a simple list of dicts. This is a local index
# by our own human-friendly appointment ID -> the underlying calendar event
# ID plus the patient details, NOT the source of truth for availability
# (the calendar is). It lets find_appointment/cancel_appointment work by ID
# or by patient details without an extra API round-trip.
#
# IMPORTANT: this index is persisted to a small JSON file on disk (see
# _load_bookings_index/_save_bookings_index below). The Google Calendar
# switch made the *appointments themselves* durable, but this index --
# which is what find_appointment/cancel_appointment actually search -- was
# still a bare in-memory list, so it was wiped on every restart exactly
# like before. Without this, a patient could book an appointment, close
# the script, reopen it, and the agent would have no way to find or cancel
# a booking that genuinely still exists on the calendar. This file is a
# cache the agent rebuilds its memory from, not a second source of truth
# for availability -- that's still always the calendar.
BOOKINGS_INDEX_FILE = Path(os.environ.get("BOOKINGS_INDEX_FILE", "bookings_index.json"))


def _load_bookings_index() -> list[dict]:
    if not BOOKINGS_INDEX_FILE.exists():
        return []
    try:
        return json.loads(BOOKINGS_INDEX_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"WARNING: could not read {BOOKINGS_INDEX_FILE} ({exc}); "
            "starting with an empty booking index. Existing calendar events "
            "are unaffected, but the agent won't be able to find/cancel them "
            "by ID or patient details until re-synced.",
            file=sys.stderr,
        )
        return []


def _save_bookings_index() -> None:
    """Called after every booking/cancellation. Best-effort: a failed save
    shouldn't crash the turn (the patient already has their confirmation),
    but it does mean this index could drift from reality until the next
    successful save, so we warn loudly rather than failing silently."""
    try:
        BOOKINGS_INDEX_FILE.write_text(
            json.dumps(BOOKINGS, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        print(f"WARNING: could not save booking index to {BOOKINGS_INDEX_FILE} ({exc}).", file=sys.stderr)


def _next_counter_start(bookings: list[dict]) -> int:
    """Resume numbering after whatever the highest ID in the loaded index
    is, so a restart never reissues an appointment ID that's already in use."""
    highest = 1000
    for booking in bookings:
        try:
            highest = max(highest, int(booking["id"].split("-")[1]))
        except (KeyError, IndexError, ValueError):
            continue
    return highest


BOOKINGS: list[dict] = _load_bookings_index()

_next_id_counter = _next_counter_start(BOOKINGS)


def _generate_appointment_id() -> str:
    global _next_id_counter
    _next_id_counter += 1
    return f"APT-{_next_id_counter}"


def _find_booking(appointment_id: str) -> dict | None:
    for booking in BOOKINGS:
        if booking["id"] == appointment_id:
            return booking
    return None


def _describe_booking(booking: dict) -> str:
    return (
        f"ID {booking['id']}: {booking['patient_name']} ({booking['phone']}), "
        f"{booking['service']}, on {booking['date']} at {booking['time']}."
    )


def _validate_date_time(date: str, time: str) -> tuple[datetime.date, dt_time] | str:
    """Parse and validate a date/time pair against the clinic's working hours.

    Returns a (date_obj, time_obj) tuple if valid, or an error message
    string explaining why it is not.
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return f"'{date}' is not a valid date. Please provide dates as YYYY-MM-DD."

    try:
        time_obj = datetime.strptime(time, "%H:%M").time()
    except ValueError:
        return f"'{time}' is not a valid time. Please provide times as HH:MM (24-hour)."

    hours = WORKING_HOURS.get(date_obj.weekday())
    if hours is None:
        return (
            f"The clinic is closed on {date_obj.strftime('%A')}s. "
            f"We're open Monday-Friday 9:00-19:00 and Saturday 9:00-14:00."
        )

    opens_at, closes_at = hours
    if not (opens_at <= time_obj < closes_at):
        return (
            f"{time} on {date_obj.strftime('%A')} {date} is outside working hours "
            f"({opens_at.strftime('%H:%M')}-{closes_at.strftime('%H:%M')} that day)."
        )

    return date_obj, time_obj


# ---------------------------------------------------------------------------
# Booking Agent — tools
# ---------------------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Get the current real-world date, day of the week, and time in the
    clinic's timezone (Europe/Madrid, Barcelona).

    Call this whenever the patient uses a relative date or time reference —
    "today", "tomorrow", "this week", "next Saturday", "in two weeks", etc.
    Your own sense of "today" is not reliable (it may be based on outdated
    training data), so always ground yourself with this tool before doing
    any date math, and before calling get_available_slots or
    book_appointment with a computed date.

    Returns:
        A string with the current weekday, date (YYYY-MM-DD), and time.
    """
    now = datetime.now(CLINIC_TIMEZONE)
    return (
        f"Current date and time in Barcelona: {now.strftime('%A')}, "
        f"{now.strftime('%Y-%m-%d')}, {now.strftime('%H:%M')} "
        f"({now.tzname()})."
    )


def _candidate_slot_starts(date_obj) -> list[datetime]:
    """Every SLOT_MINUTES-spaced slot start time that fits within working
    hours on this date (an empty list if the clinic is closed that day)."""
    hours = WORKING_HOURS.get(date_obj.weekday())
    if hours is None:
        return []
    opens_at, closes_at = hours
    day_end = datetime.combine(date_obj, closes_at, tzinfo=CLINIC_TIMEZONE)
    current = datetime.combine(date_obj, opens_at, tzinfo=CLINIC_TIMEZONE)
    slots = []
    while current + timedelta(minutes=SLOT_MINUTES) <= day_end:
        slots.append(current)
        current += timedelta(minutes=SLOT_MINUTES)
    return slots


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


@tool
def get_available_slots(date: str) -> str:
    """Get the list of available appointment time slots for a given date.

    Checks the clinic's real calendar for that day and returns whichever
    SLOT_MINUTES-long slots (within working hours) aren't already busy.

    Args:
        date: The date to check, formatted as YYYY-MM-DD.

    Returns:
        A message listing the open time slots for that date, or a message
        saying none are available / the clinic is closed / the calendar is
        temporarily unreachable.
    """
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return f"'{date}' is not a valid date. Please provide dates as YYYY-MM-DD."

    candidates = _candidate_slot_starts(date_obj)
    if not candidates:
        return (
            f"The clinic is closed on {date_obj.strftime('%A')}s. "
            f"We're open Monday-Friday 9:00-19:00 and Saturday 9:00-14:00."
        )

    try:
        busy = calendar_client.get_busy_intervals(date_obj)
    except Exception as exc:  # noqa: BLE001 -- surface as a patient-facing message, don't crash the turn
        return (
            f"Sorry, I'm having trouble reaching the booking calendar right now ({exc}). "
            f"Please try again in a moment, or call the clinic directly at {CLINIC_PHONE}."
        )

    open_slots = [
        slot_start.strftime("%H:%M")
        for slot_start in candidates
        if not any(
            _overlaps(slot_start, slot_start + timedelta(minutes=SLOT_MINUTES), b.start, b.end)
            for b in busy
        )
    ]

    if not open_slots:
        return f"There are no available slots on {date}. Please suggest another date."
    return f"Available slots on {date}: {', '.join(open_slots)}."


@tool
def book_appointment(
    patient_name: str,
    phone: str,
    date: str,
    time: str,
    service: str = "Not specified",
) -> str:
    """Book a dental appointment for a patient at a specific date and time.

    Args:
        patient_name: Full name of the patient.
        phone: Patient's contact phone number.
        date: Appointment date, formatted as YYYY-MM-DD.
        time: Appointment time, formatted as HH:MM (24-hour).
        service: The requested service, e.g. "Dental cleaning", "General
            checkup", "Filling", "Extraction", or "Orthodontics consultation".

    Returns:
        A confirmation message including the new appointment ID, or an
        error message if the request is invalid, outside working hours, the
        slot is already taken, or the calendar is temporarily unreachable.
    """
    validation = _validate_date_time(date, time)
    if isinstance(validation, str):
        return validation
    date_obj, time_obj = validation

    start = datetime.combine(date_obj, time_obj, tzinfo=CLINIC_TIMEZONE)
    end = start + timedelta(minutes=SLOT_MINUTES)

    try:
        busy = calendar_client.get_busy_intervals(date_obj)
        if any(_overlaps(start, end, b.start, b.end) for b in busy):
            return (
                f"Sorry, {time} on {date} is already booked. "
                f"Please call get_available_slots for {date} to see other open times."
            )
        event_id = calendar_client.create_event(
            summary=f"{service} - {patient_name}",
            description=(
                f"Patient: {patient_name}\nPhone: {phone}\nService: {service}\n"
                f"Booked via Laia (virtual receptionist)."
            ),
            start=start,
            end=end,
        )
    except Exception as exc:  # noqa: BLE001 -- never silently pretend a failed booking succeeded
        return (
            f"Sorry, I couldn't complete the booking because the calendar system is "
            f"unavailable right now ({exc}). Please try again shortly, or call the "
            f"clinic directly at {CLINIC_PHONE} to book by phone."
        )

    appointment_id = _generate_appointment_id()
    BOOKINGS.append(
        {
            "id": appointment_id,
            "event_id": event_id,
            "patient_name": patient_name,
            "phone": phone,
            "date": date,
            "time": time,
            "service": service,
        }
    )
    _save_bookings_index()

    return (
        f"Appointment confirmed! ID {appointment_id} for {patient_name} "
        f"({service}) on {date} at {time}. A confirmation will be sent to {phone}."
    )


@tool
def find_appointment(
    appointment_id: str = "",
    patient_name: str = "",
    phone: str = "",
    date: str = "",
    time: str = "",
) -> str:
    """Look up an existing appointment before cancelling or rescheduling it.

    Use this whenever a patient wants to cancel or reschedule but doesn't
    give you the appointment ID directly. You can search by:
      - appointment_id alone (sufficient on its own, since it's a private
        reference the patient already has), OR
      - at least TWO of: patient_name, phone, or (date AND time together,
        since either alone rarely identifies one person).

    This two-piece-of-information rule protects patient privacy — never
    treat a single detail (e.g. just a first name, or just a date) as
    enough to look up or reveal someone's appointment. If the patient has
    only given you one piece of information, call this tool anyway with
    what you have; it will tell you exactly what additional detail to ask
    for, and you should relay that request to the patient rather than
    guessing or proceeding.

    Args:
        appointment_id: Exact appointment ID if the patient has it, e.g. "APT-1001".
        patient_name: Patient's full name (or partial name) as given by them.
        phone: Patient's phone number as given by them.
        date: Appointment date, YYYY-MM-DD, if the patient remembers it.
        time: Appointment time, HH:MM, if the patient remembers it.

    Returns:
        The matching appointment's details, a request for one more piece of
        identifying information, or a "not found" message.
    """
    if appointment_id.strip():
        booking = _find_booking(appointment_id.strip())
        if booking is None:
            return f"No appointment found with ID {appointment_id}."
        return _describe_booking(booking)

    given_labels = []
    if patient_name.strip():
        given_labels.append("full name")
    if phone.strip():
        given_labels.append("phone number")
    if date.strip() and time.strip():
        given_labels.append("appointment date and time")

    if len(given_labels) < 2:
        all_labels = ["full name", "phone number", "appointment date and time"]
        missing = [label for label in all_labels if label not in given_labels]
        have_txt = f" So far I have: {', '.join(given_labels)}." if given_labels else ""
        note = ""
        if (date.strip() and not time.strip()) or (time.strip() and not date.strip()):
            note = " (Note: I need both the date and the time together for that to count.)"
        return (
            "To protect patient privacy, I need at least two pieces of "
            f"identifying information to look up an appointment.{have_txt}{note} "
            f"Could you also provide the {' or the '.join(missing)}?"
        )

    matches = []
    for b in BOOKINGS:
        if patient_name.strip() and patient_name.strip().lower() not in b["patient_name"].lower():
            continue
        if phone.strip() and phone.strip() != b["phone"]:
            continue
        if date.strip() and date.strip() != b["date"]:
            continue
        if time.strip() and time.strip() != b["time"]:
            continue
        matches.append(b)

    if not matches:
        return "No appointment found matching those details. Please double-check the information with the patient."

    if len(matches) > 1:
        listed = "\n".join(_describe_booking(b) for b in matches)
        return f"Multiple appointments matched those details:\n{listed}\nPlease ask the patient for the appointment ID to proceed."

    return _describe_booking(matches[0])


@tool
def cancel_appointment(appointment_id: str) -> str:
    """Cancel an existing dental appointment by its appointment ID.

    Args:
        appointment_id: The ID of the appointment to cancel (e.g. "APT-1001").

    Returns:
        A confirmation message if cancelled, an error message if the
        appointment ID was not found, or a message asking the patient to
        call the clinic if the calendar is temporarily unreachable.
    """
    booking = _find_booking(appointment_id)
    if booking is None:
        return f"No appointment found with ID {appointment_id}."

    try:
        calendar_client.delete_event(booking["event_id"])
    except Exception as exc:  # noqa: BLE001 -- don't drop the local record if the calendar delete failed
        return (
            f"I found appointment {appointment_id} but couldn't cancel it in the "
            f"calendar system right now ({exc}). Please call the clinic directly "
            f"at {CLINIC_PHONE} to cancel it."
        )

    BOOKINGS.remove(booking)
    _save_bookings_index()

    return (
        f"Appointment {appointment_id} for {booking['patient_name']} "
        f"on {booking['date']} at {booking['time']} has been cancelled."
    )


BOOKING_TOOLS = [
    get_current_datetime,
    get_available_slots,
    book_appointment,
    find_appointment,
    cancel_appointment,
]

# ---------------------------------------------------------------------------
# FAQ Agent — tools
# ---------------------------------------------------------------------------

@tool
def search_clinic_faq(query: str) -> str:
    """Search the clinic's general-information knowledge base.

    Covers: services offered, prices, location/address, parking, and
    accepted insurance. Always call this rather than guessing or inventing
    clinic details — if it returns nothing relevant, tell the patient you
    don't have that information and give them the clinic phone number.

    Args:
        query: The patient's question or topic, in your own words (e.g.
            "do you take Sanitas insurance", "where are you located").

    Returns:
        The matching answer(s) from the knowledge base, or a message saying
        nothing matched.
    """
    query_lower = query.lower()
    matches = [entry for entry in FAQ_KB if any(kw in query_lower for kw in entry["keywords"])]

    if not matches:
        topics = ", ".join(entry["topic"] for entry in FAQ_KB)
        return (
            "Nothing in the clinic FAQ knowledge base matches that query. "
            f"Topics available: {topics}. For anything else, direct the "
            f"patient to call the clinic directly at {CLINIC_PHONE}."
        )

    return "\n\n".join(entry["answer"] for entry in matches)


FAQ_TOOLS = [search_clinic_faq]

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

# Set ANTHROPIC_MODEL in your environment to override; update the default
# below to whichever current Claude model ID your account has access to.
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

# USD per 1,000,000 tokens. Source: https://platform.claude.com/docs/en/about-claude/pricing
# (checked 2026-09-16) -- verify against that page before trusting cost
# numbers for anything beyond rough tracking, since pricing changes.
# Matched by prefix since exact dated model IDs (e.g. "-20250929") vary.
PRICING_PER_MILLION_TOKENS = {
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


def estimate_cost_usd(usage_records: list[dict]) -> float | None:
    """Estimate USD cost from a list of usage records (see _usage_record).
    Returns None -- rather than a silently wrong number -- if MODEL_NAME
    isn't in the pricing table above."""
    pricing = next(
        (rates for prefix, rates in PRICING_PER_MILLION_TOKENS.items() if MODEL_NAME.startswith(prefix)),
        None,
    )
    if pricing is None:
        return None
    total_input = sum(r["input_tokens"] for r in usage_records)
    total_output = sum(r["output_tokens"] for r in usage_records)
    return (total_input / 1_000_000) * pricing["input"] + (total_output / 1_000_000) * pricing["output"]


llm = ChatAnthropic(model=MODEL_NAME, temperature=0)

booking_llm_with_tools = llm.bind_tools(BOOKING_TOOLS)
faq_llm_with_tools = llm.bind_tools(FAQ_TOOLS)


class RouteDecision(BaseModel):
    """The supervisor's routing decision for the next step."""
    next: Literal["booking", "faq", "FINISH"]


# include_raw=True keeps access to the underlying AIMessage (and its token
# usage) alongside the parsed RouteDecision -- without it, with_structured_
# output() would silently discard the raw response and we'd have no way to
# account for the router's token cost, which runs on every single step.
router_llm = llm.with_structured_output(RouteDecision, include_raw=True)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = f"""You are the internal routing supervisor for
{CLINIC_NAME}'s virtual receptionist system. You never speak to the patient
directly and you never answer their question yourself — you only decide, at
each step, which specialist should act next.

Specialists available:
- "booking": handles checking availability, new bookings, cancellations,
  and rescheduling. Route here for anything about getting, changing, or
  cancelling an appointment, or when the patient is giving booking details
  (name, phone, date, time, service).
- "faq": handles general questions about the clinic itself (services
  offered, prices, location/address, parking, accepted insurance), plus
  greetings and general small talk.
- "FINISH": use this once the patient's current message has been fully
  addressed and there's nothing left for another specialist to do this turn.

How to decide -- work through this every time, even if it feels repetitive:

1. Read the most recent HUMAN message and break it into its distinct
   sub-requests. Most messages have just one (e.g. "book me a cleaning
   tomorrow" = one booking sub-request). Some have two of different KINDS,
   one for each specialist (e.g. "what's the price of a cleaning, and can
   you book me one tomorrow?" = one faq sub-request about price, PLUS one
   booking sub-request). Treat each kind separately.
2. For EACH sub-request you identified, check the messages added since
   that human message: has THAT specific sub-request actually been
   substantively answered anywhere -- a real price, a real availability
   check, a real confirmation, a real clarifying question about it? Do
   this by re-reading what was actually said, not by assuming a specialist
   "must have" covered something just because it replied. A specialist
   that only answers the booking half and says nothing at all about price
   has NOT addressed the price sub-request, even if it didn't explicitly
   say "I can't answer that" -- silence on a sub-request means it's still
   unaddressed.
3. If any sub-request identified in step 1 has not been substantively
   addressed per step 2, route to whichever specialist ("booking" or
   "faq") handles that sub-request -- regardless of whether the specialist
   that just ran explicitly flagged the gap or not.
4. Only once every sub-request from step 1 has been substantively
   addressed (including the case where a specialist's reply is itself a
   necessary follow-up question, like asking for a name and phone number
   to complete a booking that was the ONLY sub-request), output "FINISH".
   A clarifying question, a "we're closed that day, how about X or Y
   instead?", or any other reply that directly and completely responds to
   the sub-request DOES count as substantively addressed -- "addressed"
   does NOT mean "the appointment is now booked" or "the conversation
   feels finished." Once a sub-request has a real, on-topic reply, it is
   done, even if the outcome was a question back to the patient or a
   polite refusal.
5. Never route to the same specialist twice in a row for a sub-request it
   has already substantively addressed -- that would be a loop, not
   progress. If you're ever unsure whether looping would help, choose
   "FINISH" rather than risk it.
6. Count the DISTINCT sub-requests from step 1, not the number of
   specialists that exist. If step 1 identified exactly ONE sub-request,
   you will visit AT MOST ONE specialist this turn no matter what --
   never route to a second, different specialist "just in case" once that
   one sub-request has a substantive reply. Only a message with genuinely
   TWO different-kind sub-requests (e.g. a price question AND a booking
   request in the same message) ever justifies visiting both "booking"
   and "faq" in the same turn.
7. If the message is a greeting, thanks, goodbye, or anything with no
   identifiable sub-request at all, route to "faq" (it handles small talk).

Only output the routing decision.
"""

BOOKING_SYSTEM_PROMPT = f"""You are Laia, the virtual receptionist for
{CLINIC_NAME}, a dental clinic in Barcelona, Spain. You are currently acting
as the Booking specialist: handling availability checks, new bookings,
cancellations, and rescheduling.

## Persona & language
Warm, professional, and efficient. Reply in whichever language the patient
is using (Spanish, Catalan, or English), matching them if they switch
mid-conversation, without commenting on the switch.

## Clinic information you may need
Services offered: {", ".join(SERVICES)}.
Working hours: Monday-Friday 9:00-19:00, Saturday 9:00-14:00, closed Sundays
and public holidays.
Clinic phone number (for anything you can't handle, or emergencies): {CLINIC_PHONE}.

## Rules — always follow these
1. Never give medical advice, diagnoses, or treatment recommendations of any
   kind, and never suggest remedies (even simple ones like "rinse with salt
   water"). You handle scheduling only.
2. If the patient mentions pain, a toothache, swelling, bleeding, trauma, a
   broken tooth, or anything that sounds like a dental emergency, do not
   route them through normal booking. Tell them plainly to call the clinic
   directly at {CLINIC_PHONE} for urgent attention, or, if it sounds like a
   medical emergency, to go to the nearest emergency room. Let them know
   you're escalating this to a human member of staff rather than booking it.
3. Whenever the patient uses any relative date or time reference — "today",
   "tomorrow", "this week", "next Saturday", "in two weeks", etc. — call
   get_current_datetime first, then work out the exact YYYY-MM-DD date
   yourself before calling any other tool. Never guess today's date, and
   never pass a relative expression directly to a tool. Treat the week as
   Monday through Sunday.
4. If a patient asks for a Sunday appointment (or any closed day), explain
   the clinic is closed that day and proactively suggest the nearest open
   days (e.g. the Saturday before or the Monday after).
5. Before calling book_appointment, always read back and confirm: patient's
   full name, phone number, date, time, and requested service. Get explicit
   confirmation before booking.
6. Always call get_available_slots to check real availability before
   proposing a time — never invent availability.
7. Always call book_appointment to actually create a booking, and relay its
   result (including the appointment ID) plainly. If it reports the slot is
   outside working hours or already taken, explain that and help the
   patient pick another time.
8. To cancel or reschedule, first identify the right appointment. If the
   patient gives the appointment ID, use it. Otherwise call find_appointment
   with whatever details they've given (name, phone, date+time). If it asks
   for one more piece of information, request that from the patient before
   proceeding — never cancel/modify an appointment you haven't positively
   identified, and never reveal details to someone with only one piece of
   identifying information.
9. To cancel: once identified and confirmed, call cancel_appointment.
10. To reschedule (cancel-then-rebook): identify the existing appointment,
    confirm the new date/time is available via get_available_slots, confirm
    the change with the patient (old slot cancelled, new slot booked), then
    call cancel_appointment on the old ID and book_appointment for the new
    slot, re-using the same name/phone/service unless told otherwise. Tell
    the patient both that the old appointment was cancelled and the new
    appointment's ID.
11. If the patient's message also includes something outside booking (e.g.
    a pricing or location question), handle the booking part and always
    say a brief line acknowledging the other part too (e.g. "I'll also get
    you the pricing on that") — never answer only the booking part in
    total silence about the rest, even though the system also independently
    tracks and routes unaddressed parts on its own.
12. Keep responses concise and natural, like a receptionist speaking.
"""

FAQ_SYSTEM_PROMPT = f"""You are Laia, the virtual receptionist for
{CLINIC_NAME}, a dental clinic in Barcelona, Spain. You are currently acting
as the FAQ specialist: answering general questions about the clinic, and
handling greetings/small talk.

## Persona & language
Warm, professional, and efficient. Reply in whichever language the patient
is using (Spanish, Catalan, or English), matching them if they switch
mid-conversation, without commenting on the switch.

## Your job here
Answer general questions about the clinic — services offered, prices,
location/address, parking, and accepted insurance — by calling
search_clinic_faq. Always call that tool rather than guessing or making up
clinic details; if it doesn't have the answer, tell the patient you don't
have that detail and give them the clinic phone number: {CLINIC_PHONE}.

If the patient is just greeting you, saying thanks, or making small talk,
respond warmly and briefly and ask how you can help — no tool call needed
for that.

## Boundaries
- Never give medical advice, diagnoses, or treatment recommendations.
- If the patient mentions pain, a toothache, swelling, bleeding, trauma, or
  anything sounding like a dental emergency, do not try to answer it
  yourself — tell them plainly to call the clinic directly at {CLINIC_PHONE}
  for urgent attention, or go to the nearest emergency room if serious.
- If the patient wants to check availability, book, cancel, or reschedule,
  briefly acknowledge it — you don't need to do anything else, the system
  routes that automatically.
- Keep responses concise and natural, like a receptionist speaking.
"""

# ---------------------------------------------------------------------------
# Specialist sub-agents — each a standard LangGraph ReAct subgraph
# ---------------------------------------------------------------------------

def _make_agent_node(llm_with_tools):
    """Build an 'agent' node function bound to a specific specialist LLM."""
    def _agent_node(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    return _agent_node


def _build_specialist_subgraph(tools: list, llm_with_tools):
    """Standard ReAct pattern: agent -> tools_condition -> tools -> agent -> ... -> END."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", _make_agent_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()  # no checkpointer: run fresh each time on the full history it's given


booking_subgraph = _build_specialist_subgraph(BOOKING_TOOLS, booking_llm_with_tools)
faq_subgraph = _build_specialist_subgraph(FAQ_TOOLS, faq_llm_with_tools)


# ---------------------------------------------------------------------------
# Top-level graph: Supervisor -> {Booking, FAQ} -> back to Supervisor
# ---------------------------------------------------------------------------

# Hard cap on how many specialists can run per single patient turn. This is
# a code-level safety net, not just a prompt instruction: an LLM classifier
# can occasionally misjudge "should I route again or finish?", and without
# this cap a bad judgment call could bounce between supervisor and a
# specialist many times — each bounce costing at least one extra API call —
# before finally hitting LangGraph's recursion limit. Two hops is enough to
# cover a genuinely mixed request (one visit to booking, one to faq) while
# making runaway loops impossible.
MAX_SPECIALIST_HOPS_PER_TURN = 2


def _usage_record(agent_name: str, ai_message) -> dict | None:
    """Pull standardized token-usage info off an AIMessage, if present.

    langchain_core populates `.usage_metadata` on AIMessages that come from
    a real model call (input_tokens/output_tokens/total_tokens). It's absent
    on messages from stub/fake LLMs (as used in offline tests), so this
    returns None rather than raising -- callers should skip None results.
    """
    usage = getattr(ai_message, "usage_metadata", None)
    if not usage:
        return None
    return {
        "agent": agent_name,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _usage_records(agent_name: str, messages: list) -> list[dict]:
    from langchain_core.messages import AIMessage
    records = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            record = _usage_record(agent_name, msg)
            if record is not None:
                records.append(record)
    return records


class SupervisorState(MessagesState):
    """Shared state: the conversation, the routing decision, a per-turn hop
    counter that hard-caps how many specialists can run in one turn, and a
    running log of token usage per LLM call (for cost/latency reporting --
    see run_evals.py). `usage` accumulates via list concatenation (the
    `operator.add` reducer) rather than being overwritten each step."""
    next: str
    specialist_hops: int
    usage: Annotated[list[dict], operator.add]


def supervisor_node(state: SupervisorState):
    """Classify intent and decide which specialist (if any) acts next."""
    if state.get("specialist_hops", 0) >= MAX_SPECIALIST_HOPS_PER_TURN:
        # Safety net: force the turn to end rather than risk another loop.
        # (No LLM call here, so nothing to record in usage.)
        return {"next": "FINISH"}
    response = router_llm.invoke([SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"])
    decision: RouteDecision = response["parsed"]
    usage_record = _usage_record("supervisor", response["raw"])
    return {"next": decision.next, "usage": [usage_record] if usage_record else []}


def booking_node(state: SupervisorState):
    """Run the Booking sub-agent on the full conversation, with its own system prompt."""
    input_messages = [SystemMessage(content=BOOKING_SYSTEM_PROMPT)] + state["messages"]
    result = booking_subgraph.invoke({"messages": input_messages})
    new_messages = result["messages"][len(input_messages):]  # only what the sub-agent added
    return {
        "messages": new_messages,
        "specialist_hops": state.get("specialist_hops", 0) + 1,
        "usage": _usage_records("booking", new_messages),
    }


def faq_node(state: SupervisorState):
    """Run the FAQ sub-agent on the full conversation, with its own system prompt."""
    input_messages = [SystemMessage(content=FAQ_SYSTEM_PROMPT)] + state["messages"]
    result = faq_subgraph.invoke({"messages": input_messages})
    new_messages = result["messages"][len(input_messages):]  # only what the sub-agent added
    return {
        "messages": new_messages,
        "specialist_hops": state.get("specialist_hops", 0) + 1,
        "usage": _usage_records("faq", new_messages),
    }


def build_graph():
    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("booking", booking_node)
    workflow.add_node("faq", faq_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"booking": "booking", "faq": "faq", "FINISH": END},
    )

    # After a specialist finishes, control returns to the supervisor, which
    # decides whether to route again or finish this turn.
    workflow.add_edge("booking", "supervisor")
    workflow.add_edge("faq", "supervisor")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# CLI chat loop
# ---------------------------------------------------------------------------

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: Set the ANTHROPIC_API_KEY environment variable "
            "(or put it in a .env file) before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    graph = build_graph()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Every session gets its own readable transcript file: one JSON object
    # per exchange, so you can open it later and see exactly what was said,
    # how long each turn took, and (when usage data is available) its
    # estimated cost -- without needing to log into any external platform.
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(exist_ok=True)
    session_start = datetime.now(CLINIC_TIMEZONE)
    transcript_path = transcripts_dir / f"{session_start.strftime('%Y-%m-%d_%H%M%S')}_{thread_id[:8]}.jsonl"

    print(f"{CLINIC_NAME} — Appointment Assistant (multi-agent)")
    print("Type 'exit' or 'quit' to end the chat.\n")
    print(f"(Transcript being saved to {transcript_path})\n")

    usage_seen_so_far = 0  # tracks how many usage records we've already logged

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print(f"Laia: Thanks for contacting {CLINIC_NAME}. Have a great day!")
            break

        if not user_input:
            continue

        turn_started_at = time.perf_counter()
        try:
            result = graph.invoke(
                # Reset the per-turn hop counter on every new patient message.
                {"messages": [HumanMessage(content=user_input)], "specialist_hops": 0},
                config={**config, "recursion_limit": 12},
            )
        except GraphRecursionError:
            print(
                "Laia: Sorry, I got a bit tangled up processing that. "
                "Could you rephrase or split that into separate questions?\n"
            )
            continue
        latency_seconds = round(time.perf_counter() - turn_started_at, 2)

        # Only this turn's usage records (usage accumulates all session-long,
        # so slice off what's new since the last turn).
        all_usage = result.get("usage", [])
        turn_usage = all_usage[usage_seen_so_far:]
        usage_seen_so_far = len(all_usage)
        turn_cost = estimate_cost_usd(turn_usage)

        reply_text = last_reply_text(result["messages"])

        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(CLINIC_TIMEZONE).isoformat(),
                "user": user_input,
                "assistant": reply_text,
                "specialist_hops": result.get("specialist_hops", 0),
                "latency_seconds": latency_seconds,
                "usage": turn_usage,
                "estimated_cost_usd": turn_cost,
            }, ensure_ascii=False) + "\n")

        print(f"Laia: {reply_text}\n")


if __name__ == "__main__":
    main()
