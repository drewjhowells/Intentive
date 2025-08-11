"""
calendar_store.py — minimal SQLite calendar for Alden agents

Purpose:
    Provide a super simple calendar until the frontend is ready.
    Agents can create, read (by window), update, and delete events.

Storage:
    stores/calendar.db (SQLite)

Timestamps:
    Use ISO 8601 UTC strings, e.g. "2025-08-11T15:30:00Z"

Notes:
    - Keep it simple now; we can add recurrence/attendees later.
    - All queries scoped to a "calendar" name; default = "default".
"""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), "stores", "calendar.db")


# ---------- Helpers ----------

def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with 'Z'."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _ensure_db():
    """Create DB and table if missing."""
    os.makedirs(os.path.join(os.path.dirname(__file__), "stores"), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            calendar TEXT NOT NULL DEFAULT 'default',
            title TEXT NOT NULL,
            start_utc TEXT NOT NULL,       -- ISO 8601 UTC (e.g., 2025-08-11T09:00:00Z)
            end_utc   TEXT NOT NULL,       -- must be > start_utc
            all_day   INTEGER NOT NULL DEFAULT 0,  -- 0/1
            location  TEXT,
            notes     TEXT,
            status    TEXT DEFAULT 'confirmed',    -- e.g., tentative/confirmed/canceled
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(start_utc, end_utc);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_events_calendar ON events(calendar);")
    con.commit()
    con.close()

def _connect():
    """Open a new connection."""
    _ensure_db()
    return sqlite3.connect(DB_PATH)


# ---------- CRUD API ----------

def create_event(
    title: str,
    start_utc: str,
    end_utc: str,
    calendar: str = "default",
    all_day: bool = False,
    location: Optional[str] = None,
    notes: Optional[str] = None,
    status: str = "confirmed",
) -> int:
    """
    Create an event and return its ID.

    Args:
        title: Short name of the event
        start_utc/end_utc: ISO 8601 UTC strings (e.g., '2025-08-11T09:00:00Z')
        calendar: Logical calendar name
        all_day: If True, treat as all-day (start/end still stored)
        location/notes/status: Optional metadata
    """
    _ensure_db()
    if end_utc <= start_utc:
        raise ValueError("end_utc must be after start_utc")

    now = _utc_now_iso()
    con = _connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO events (calendar, title, start_utc, end_utc, all_day, location, notes, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (calendar, title, start_utc, end_utc, int(all_day), location, notes, status, now, now))
    con.commit()
    event_id = cur.lastrowid
    con.close()
    return event_id


def get_events_window(
    back_days: int = 0,
    forward_days: int = 7,
    calendar: str = "default",
    now_utc_iso: Optional[str] = None
) -> List[Dict]:
    """
    Return events in a time window relative to 'now'.

    Args:
        back_days: How many days back to include
        forward_days: How many days forward to include
        calendar: Which calendar to query
        now_utc_iso: Override 'now' (ISO 8601 UTC). If None, uses current UTC.

    Returns:
        List of event dicts sorted by start_utc.
    """
    _ensure_db()
    now = datetime.fromisoformat(now_utc_iso.replace("Z", "+00:00")) if now_utc_iso else datetime.now(timezone.utc)
    start_window = (now - timedelta(days=back_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    end_window   = (now + timedelta(days=forward_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    con = _connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
        SELECT * FROM events
        WHERE calendar = ?
          AND (
                (start_utc <= ? AND end_utc >= ?)  -- spanning now
             OR (start_utc >= ? AND start_utc <= ?) -- starting within window
             OR (end_utc   >= ? AND end_utc   <= ?) -- ending within window
          )
        ORDER BY start_utc ASC
    """, (calendar, end_window, start_window, start_window, end_window, start_window, end_window))
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_event(event_id: int) -> Optional[Dict]:
    """Fetch a single event by ID."""
    _ensure_db()
    con = _connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else None


def update_event(event_id: int, **fields) -> bool:
    """
    Update an event by ID. Allowed fields: title, start_utc, end_utc,
    calendar, all_day, location, notes, status.

    Returns:
        True if updated (row existed), False otherwise.
    """
    _ensure_db()
    if not fields:
        return False

    allowed = {"title", "start_utc", "end_utc", "calendar", "all_day", "location", "notes", "status"}
    set_parts: List[str] = []
    values: List = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "all_day":
            v = int(bool(v))
        set_parts.append(f"{k} = ?")
        values.append(v)

    if not set_parts:
        return False

    values.append(_utc_now_iso())
    values.append(event_id)

    con = _connect()
    cur = con.cursor()
    sql = f"UPDATE events SET {', '.join(set_parts)}, updated_at = ? WHERE id = ?"
    cur.execute(sql, values)
    con.commit()
    changed = cur.rowcount > 0
    con.close()
    return changed


def delete_event(event_id: int) -> bool:
    """Delete an event by ID. Returns True if a row was removed."""
    _ensure_db()
    con = _connect()
    cur = con.cursor()
    cur.execute("DELETE FROM events WHERE id = ?", (event_id,))
    con.commit()
    changed = cur.rowcount > 0
    con.close()
    return changed


# ---------- Convenience: quick add & sample run ----------

def quick_add_block(
    title: str,
    start_in_minutes: int,
    duration_minutes: int,
    calendar: str = "default",
    notes: Optional[str] = None
) -> int:
    """
    Create an event starting N minutes from now for M minutes.
    Handy for agent-generated blocks (prep/focus/wrap).

    Returns: event_id
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now + timedelta(minutes=start_in_minutes)
    end = start + timedelta(minutes=duration_minutes)
    return create_event(
        title=title,
        start_utc=start.isoformat().replace("+00:00", "Z"),
        end_utc=end.isoformat().replace("+00:00", "Z"),
        calendar=calendar,
        notes=notes
    )


if __name__ == "__main__":
    # Minimal smoke test
    _ensure_db()
    eid = quick_add_block("Practice Block", start_in_minutes=5, duration_minutes=30, notes="Created by __main__")
    print("Created event:", eid)
    window = get_events_window(back_days=0, forward_days=1)
    print("Upcoming events (1 day):", window)
    updated = update_event(eid, title="Practice Block (updated)")
    print("Updated:", updated, "->", get_event(eid))
    deleted = delete_event(eid)
    print("Deleted:", deleted)