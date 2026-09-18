"""
One-off manual smoke test for the REAL Google Calendar integration -- run
this once right after you've done the setup in calendar_client.py's
docstring (service account, shared calendar, GOOGLE_SERVICE_ACCOUNT_FILE /
GOOGLE_CALENDAR_ID set), before trusting the full agent with it.

This is deliberately NOT a pytest test: it needs real credentials and a
real network call, and it creates (then deletes) a real event on your
actual calendar. Run it by hand, read the output, and only move on once
every step prints "OK".

Usage:
    venv\\Scripts\\python.exe calendar_smoke_test.py

What it checks, in order:
    1. Credentials load and the client builds without error.
    2. A busy-time query for tomorrow succeeds (proves read access + the
       calendar is actually shared with the service account).
    3. Creating an event succeeds, and that new event shows up as "busy"
       when queried again (proves write access AND that busy-time
       computation actually reflects real events, not just that the API
       call didn't error).
    4. Deleting that event succeeds, and it disappears from the busy query
       (proves cleanup works, so this script doesn't leave test junk on
       your calendar).

If any step fails, the printed error is usually one of:
    - "File not found" on the service account file -> check
      GOOGLE_SERVICE_ACCOUNT_FILE's path.
    - A 404 on the calendar ID -> check GOOGLE_CALENDAR_ID, and that you
      copied the exact ID from the calendar's Settings > "Integrate
      calendar" section.
    - A 403 "Forbidden" -> the calendar isn't actually shared with the
      service account's email yet, or was shared with only "See only
      free/busy" instead of "Make changes to events".
"""

import os
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()


def main():
    service_account_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
    calendar_id = os.environ.get("GOOGLE_CALENDAR_ID")
    if not service_account_file or not calendar_id:
        print(
            "ERROR: set GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_CALENDAR_ID first "
            "(see calendar_client.py's docstring for setup steps).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import here (not at module level) so dental_agent.py's own CLINIC_TIMEZONE
    # / config doesn't need to be duplicated -- we just need the same timezone.
    import dental_agent as agent
    from calendar_client import GoogleCalendarClient

    print(f"1. Building GoogleCalendarClient for calendar '{calendar_id}' ...")
    client = GoogleCalendarClient(calendar_id, service_account_file, agent.CLINIC_TIMEZONE)
    print("   OK\n")

    tomorrow = (datetime.now(agent.CLINIC_TIMEZONE) + timedelta(days=1)).date()

    print(f"2. Querying busy intervals for {tomorrow} (before creating anything) ...")
    busy_before = client.get_busy_intervals(tomorrow)
    print(f"   OK -- {len(busy_before)} existing busy interval(s) found.\n")

    print("3. Creating a test event ...")
    start = datetime.combine(tomorrow, agent.dt_time(10, 0), tzinfo=agent.CLINIC_TIMEZONE)
    end = start + timedelta(minutes=30)
    event_id = client.create_event(
        summary="[TEST -- safe to ignore/delete] dental_agent calendar_smoke_test.py",
        description="Created by calendar_smoke_test.py to verify write access. Will be deleted automatically.",
        start=start,
        end=end,
    )
    print(f"   OK -- created event ID {event_id}\n")

    print("4. Re-querying busy intervals to confirm the new event shows up ...")
    busy_after = client.get_busy_intervals(tomorrow)
    if len(busy_after) != len(busy_before) + 1:
        print(
            f"   WARNING: expected {len(busy_before) + 1} busy interval(s), "
            f"got {len(busy_after)}. The event was created, but the busy-time "
            f"query didn't reflect it the way we expected -- double check manually."
        )
    else:
        print("   OK -- busy interval count increased by exactly 1.\n")

    print("5. Deleting the test event ...")
    client.delete_event(event_id)
    print("   OK\n")

    print("6. Re-querying busy intervals to confirm cleanup ...")
    busy_final = client.get_busy_intervals(tomorrow)
    if len(busy_final) != len(busy_before):
        print(
            f"   WARNING: expected to be back to {len(busy_before)} busy interval(s), "
            f"got {len(busy_final)}. Check your calendar manually for leftover test events."
        )
    else:
        print("   OK -- back to the original count.\n")

    print("All steps completed. Your Google Calendar integration looks correctly configured.")


if __name__ == "__main__":
    main()
