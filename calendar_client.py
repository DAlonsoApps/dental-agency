"""
Thin abstraction over "the calendar" that the dental agent's booking tools
talk to, instead of calling Google's API (or anything else) directly.

Why bother with an interface instead of just calling Google APIs inline in
dental_agent.py? Two reasons:

  1. Testability. The offline test tiers (see test_dental_agent.py) need to
     exercise book_appointment / cancel_appointment / get_available_slots
     with zero network calls and fully deterministic results. FakeCalendarClient
     below is an in-memory stand-in that behaves like a real calendar for
     that purpose -- same pattern as the fake LLMs already used to test the
     Supervisor routing offline.
  2. Swappability. If you ever want to move off Google Calendar (a
     different provider, a real practice-management system, etc.), only
     this file needs a new CalendarClient implementation -- dental_agent.py's
     tools don't change.

Setup for real Google Calendar access (service account -- no browser login,
no per-user consent screen, works well for a single always-on clinic
calendar):

  1. Go to https://console.cloud.google.com/ and create a project (or reuse
     one you already have).
  2. In "APIs & Services" > "Library", search for "Google Calendar API" and
     click Enable.
  3. In "APIs & Services" > "Credentials", click "Create Credentials" >
     "Service account". Give it any name (e.g. "dental-agent"). You don't
     need to grant it any project-level IAM role for this.
  4. Open the service account you just created > "Keys" tab > "Add Key" >
     "Create new key" > JSON. This downloads a .json file -- save it
     somewhere safe (NOT committed to git) and note its path.
  5. Open Google Calendar in your browser. Either use an existing calendar
     or create a new one dedicated to this clinic ("Settings" > "Create new
     calendar"). Open that calendar's Settings > "Share with specific
     people", add the service account's email address (it looks like
     something@your-project.iam.gserviceaccount.com -- found in the JSON
     key file's "client_email" field, or on the service account's page),
     and give it "Make changes to events" permission.
  6. Find the Calendar ID: in that same Settings page, scroll to
     "Integrate calendar" -- the "Calendar ID" field (looks like
     abc123...@group.calendar.google.com, or your own email address if you
     shared your primary calendar).
  7. Set two environment variables (e.g. in your .env file):
         GOOGLE_SERVICE_ACCOUNT_FILE=C:\\path\\to\\service-account.json
         GOOGLE_CALENDAR_ID=abc123...@group.calendar.google.com

If those two variables aren't set, dental_agent.py automatically falls back
to FakeCalendarClient (with a warning printed) so the script and the offline
tests still run without any Google setup -- bookings just won't persist
between runs until you configure this.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, datetime, time as dt_time


@dataclass
class BusyInterval:
    start: datetime
    end: datetime


class CalendarClient:
    """Interface both the real Google client and the offline fake implement.

    All three methods should raise on failure (network error, auth error,
    not-found, etc.) rather than swallowing it -- dental_agent.py's tools
    catch exceptions here and turn them into a patient-facing message that
    asks them to try again or call the clinic, instead of silently
    pretending a booking succeeded when it didn't.
    """

    def get_busy_intervals(self, day: date_type) -> list[BusyInterval]:
        """Return every busy interval on the calendar that overlaps `day`."""
        raise NotImplementedError

    def create_event(self, *, summary: str, description: str, start: datetime, end: datetime) -> str:
        """Create a calendar event and return its event ID."""
        raise NotImplementedError

    def delete_event(self, event_id: str) -> None:
        """Delete a previously created event. Raises if it doesn't exist."""
        raise NotImplementedError


def _parse_gcal_datetime(value: str) -> datetime:
    """Google's API returns RFC3339 timestamps, sometimes with a trailing
    'Z' for UTC. datetime.fromisoformat handles 'Z' natively from Python
    3.11 onward, but we normalize it ourselves so this also works on 3.9/3.10."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


class GoogleCalendarClient(CalendarClient):
    """Real Google Calendar backend, authenticated as a service account."""

    def __init__(self, calendar_id: str, service_account_file: str, timezone):
        # Imported lazily so `pip install`-ing the Google packages is only
        # required if you actually use this class -- FakeCalendarClient
        # (and therefore all offline tests) never needs them.
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        self._service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
        self._calendar_id = calendar_id
        self._timezone = timezone

    def get_busy_intervals(self, day: date_type) -> list[BusyInterval]:
        day_start = datetime.combine(day, dt_time.min, tzinfo=self._timezone)
        day_end = datetime.combine(day, dt_time.max, tzinfo=self._timezone)
        body = {
            "timeMin": day_start.isoformat(),
            "timeMax": day_end.isoformat(),
            "items": [{"id": self._calendar_id}],
        }
        response = self._service.freebusy().query(body=body).execute()
        busy_raw = response["calendars"][self._calendar_id].get("busy", [])
        return [
            BusyInterval(
                start=_parse_gcal_datetime(entry["start"]),
                end=_parse_gcal_datetime(entry["end"]),
            )
            for entry in busy_raw
        ]

    def create_event(self, *, summary: str, description: str, start: datetime, end: datetime) -> str:
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": str(self._timezone)},
            "end": {"dateTime": end.isoformat(), "timeZone": str(self._timezone)},
        }
        created = self._service.events().insert(calendarId=self._calendar_id, body=event).execute()
        return created["id"]

    def delete_event(self, event_id: str) -> None:
        self._service.events().delete(calendarId=self._calendar_id, eventId=event_id).execute()


class FakeCalendarClient(CalendarClient):
    """In-memory stand-in for tests and for running without Google set up
    yet. No network, fully deterministic, and its `events` dict is directly
    inspectable from a test if you need to assert on what got created."""

    def __init__(self):
        self.events: dict[str, dict] = {}
        self._next_id = 1

    def get_busy_intervals(self, day: date_type) -> list[BusyInterval]:
        return [
            BusyInterval(start=event["start"], end=event["end"])
            for event in self.events.values()
            if event["start"].date() == day
        ]

    def create_event(self, *, summary: str, description: str, start: datetime, end: datetime) -> str:
        event_id = f"fake-evt-{self._next_id}"
        self._next_id += 1
        self.events[event_id] = {"summary": summary, "description": description, "start": start, "end": end}
        return event_id

    def delete_event(self, event_id: str) -> None:
        if event_id not in self.events:
            raise KeyError(f"No such event: {event_id}")
        del self.events[event_id]
