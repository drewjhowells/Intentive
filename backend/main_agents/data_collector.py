# data_collector.py
"""
Minimal Data Collector shell with category-based validators and stores.

Supported categories:
  - GPS: expects {"coords": "lat,lon"} or {"lat": float, "lon": float}
  - CALENDAR: expects {"events": [ {title,start_time,end_time,location?,description?}, ... ]}
    • times are ISO 8601 strings in UTC (e.g., "2025-08-10T14:00:00Z")
  - SCREEN_USAGE: expects {"sessions": [ {app_name,package_name,start_time,end_time?,duration_seconds?}, ... ]}
    • if end_time missing but duration provided, we store duration; if duration missing but start+end provided, we compute duration
  - USER: accepts any JSON payload and stores it as a single JSON blob for auditing/debugging

All stores are lightweight SQLite DBs created on demand under ../stores/ .
Each store logs with ts_utc (collector insert time) to keep ingestion consistent.

Notes:
- Validators are intentionally permissive right now (light checks + helpful debug prints).
- Timestamps are expected as ISO 8601; a trailing 'Z' will be normalized to +00:00.
"""
from __future__ import annotations
from typing import Dict, Any, Literal, Callable, List, Tuple
import sqlite3
from datetime import datetime, timezone
import os
import json

Category = Literal["GPS", "CALENDAR", "SCREEN_USAGE", "USER"]

# ----------------------- helpers -----------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_to_dt(s: str) -> datetime:
    """Parse ISO 8601 strings; supports trailing 'Z'. Raises ValueError if invalid."""
    if not isinstance(s, str):
        raise ValueError("timestamp must be a string")
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


# ----------------------- validators -----------------------

def _validate_gps(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("GPS payload must be an object")
    if "coords" in payload:
        if not isinstance(payload["coords"], str) or "," not in payload["coords"]:
            raise ValueError("GPS.coords must be a 'lat,lon' string")
        # basic float check
        lat_str, lon_str = payload["coords"].split(",", 1)
        float(lat_str); float(lon_str)
        return
    if "lat" in payload and "lon" in payload:
        float(payload["lat"]); float(payload["lon"])  # raises on invalid
        return
    raise ValueError("GPS payload requires either 'coords' or 'lat'+'lon'")


def _validate_calendar(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("CALENDAR payload must be an object")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("CALENDAR.events must be a list")
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise ValueError(f"CALENDAR.events[{idx}] must be an object")
        # required
        for k in ("title", "start_time", "end_time"):
            if k not in ev:
                raise ValueError(f"CALENDAR.events[{idx}] missing '{k}'")
        # time sanity
        _iso_to_dt(ev["start_time"])  # will raise on invalid
        _iso_to_dt(ev["end_time"])    # will raise on invalid


def _validate_screen_usage(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("SCREEN_USAGE payload must be an object")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("SCREEN_USAGE.sessions must be a list")
    for idx, s in enumerate(sessions):
        if not isinstance(s, dict):
            raise ValueError(f"SCREEN_USAGE.sessions[{idx}] must be an object")
        for k in ("app_name", "package_name", "start_time"):
            if k not in s:
                raise ValueError(f"SCREEN_USAGE.sessions[{idx}] missing '{k}'")
        # validate time/duration if present
        _iso_to_dt(s["start_time"])  # raises on invalid
        if "end_time" in s and s["end_time"] is not None:
            _iso_to_dt(s["end_time"])  # raises
        if "duration_seconds" in s and s["duration_seconds"] is not None:
            int(s["duration_seconds"])  # raises


def _validate_user(payload: Dict[str, Any]) -> None:
    # Intentionally permissive; we just store the JSON blob
    if not isinstance(payload, dict):
        raise ValueError("USER payload must be an object")


VALIDATORS: dict[Category, Callable[[Dict[str, Any]], None]] = {
    "GPS": _validate_gps,
    "CALENDAR": _validate_calendar,
    "SCREEN_USAGE": _validate_screen_usage,
    "USER": _validate_user,
}

# ----------------------- stores (SQLite) -----------------------
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stores"))
GPS_DB_PATH = os.path.join(_BASE, "gps.db")
CAL_DB_PATH = os.path.join(_BASE, "calendar.db")
SCREEN_DB_PATH = os.path.join(_BASE, "screen_usage.db")
USER_DB_PATH = os.path.join(_BASE, "user.db")


def init_gps_store(debug: bool = False) -> None:
    os.makedirs(_BASE, exist_ok=True)
    with sqlite3.connect(GPS_DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS gps (
                ts_utc TEXT,
                lat REAL,
                lon REAL
            )
            """
        )
    if debug:
        print(f"[INIT] GPS store initialized at {GPS_DB_PATH}")


def init_calendar_store(debug: bool = False) -> None:
    os.makedirs(_BASE, exist_ok=True)
    with sqlite3.connect(CAL_DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar (
                ts_utc TEXT,
                event_id TEXT,
                title TEXT,
                start_utc TEXT,
                end_utc TEXT,
                location TEXT,
                description TEXT
            )
            """
        )
    if debug:
        print(f"[INIT] CALENDAR store initialized at {CAL_DB_PATH}")


def init_screen_usage_store(debug: bool = False) -> None:
    os.makedirs(_BASE, exist_ok=True)
    with sqlite3.connect(SCREEN_DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS screen_usage (
                ts_utc TEXT,
                app_name TEXT,
                package_name TEXT,
                start_utc TEXT,
                end_utc TEXT,
                duration_seconds INTEGER
            )
            """
        )
    if debug:
        print(f"[INIT] SCREEN_USAGE store initialized at {SCREEN_DB_PATH}")


def init_user_store(debug: bool = False) -> None:
    os.makedirs(_BASE, exist_ok=True)
    with sqlite3.connect(USER_DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS user (
                ts_utc TEXT,
                data_json TEXT
            )
            """
        )
    if debug:
        print(f"[INIT] USER store initialized at {USER_DB_PATH}")


def _store_gps(payload: Dict[str, Any], *, debug: bool) -> None:
    """Store a single GPS reading."""
    init_gps_store(debug=False)
    ts = _utc_now_iso()
    if "coords" in payload and isinstance(payload["coords"], str):
        lat_str, lon_str = payload["coords"].split(",", 1)
        lat, lon = float(lat_str.strip()), float(lon_str.strip())
    elif "lat" in payload and "lon" in payload:
        lat, lon = float(payload["lat"]), float(payload["lon"])
    else:
        if debug:
            print("[STORE][GPS] No GPS data found in payload")
        return
    with sqlite3.connect(GPS_DB_PATH) as con:
        con.execute(
            "INSERT INTO gps (ts_utc, lat, lon) VALUES (?, ?, ?)",
            (ts, lat, lon),
        )
    if debug:
        print("[STORE][GPS] Row inserted.")


def _store_calendar(payload: Dict[str, Any], *, debug: bool) -> None:
    """Store calendar events list."""
    init_calendar_store(debug=False)
    ts = _utc_now_iso()
    events: List[Dict[str, Any]] = payload.get("events", [])
    if not events:
        if debug:
            print("[STORE][CALENDAR] No events found.")
        return
    rows: List[Tuple[str, str, str, str, str, str, str]] = []
    for ev in events:
        title = str(ev.get("title", ""))
        start = str(ev.get("start_time", ""))
        end = str(ev.get("end_time", ""))
        location = str(ev.get("location", ""))
        description = str(ev.get("description", ""))
        event_id = str(ev.get("event_id", ""))  # optional, helps with de-dup later
        rows.append((ts, event_id, title, start, end, location, description))
    with sqlite3.connect(CAL_DB_PATH) as con:
        con.executemany(
            """
            INSERT INTO calendar (ts_utc, event_id, title, start_utc, end_utc, location, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    if debug:
        print(f"[STORE][CALENDAR] Inserted {len(rows)} event(s).")


def _store_screen_usage(payload: Dict[str, Any], *, debug: bool) -> None:
    """Store screen/app usage sessions list."""
    init_screen_usage_store(debug=False)
    ts = _utc_now_iso()
    sessions: List[Dict[str, Any]] = payload.get("sessions", [])
    if not sessions:
        if debug:
            print("[STORE][SCREEN_USAGE] No sessions found.")
        return
    rows: List[Tuple[str, str, str, str, str, int]] = []
    for s in sessions:
        app_name = str(s.get("app_name", ""))
        pkg = str(s.get("package_name", ""))
        start = str(s.get("start_time", ""))
        end = s.get("end_time")
        duration = s.get("duration_seconds")
        # compute duration if needed and possible
        if duration is None and end:
            try:
                dt_start = _iso_to_dt(start)
                dt_end = _iso_to_dt(str(end))
                duration = max(0, int((dt_end - dt_start).total_seconds()))
            except Exception:
                duration = None
        if end is None and duration is not None:
            end = ""  # keep empty if unknown; we still store duration
        rows.append((ts, app_name, pkg, start, str(end) if end is not None else "", int(duration) if duration is not None else None))

    with sqlite3.connect(SCREEN_DB_PATH) as con:
        con.executemany(
            """
            INSERT INTO screen_usage (ts_utc, app_name, package_name, start_utc, end_utc, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    if debug:
        print(f"[STORE][SCREEN_USAGE] Inserted {len(rows)} session(s).")


def _store_user(payload: Dict[str, Any], *, debug: bool) -> None:
    """Store an opaque USER payload as a JSON blob for auditing/debugging."""
    init_user_store(debug=False)
    ts = _utc_now_iso()
    blob = json.dumps(payload, ensure_ascii=False)
    with sqlite3.connect(USER_DB_PATH) as con:
        con.execute(
            "INSERT INTO user (ts_utc, data_json) VALUES (?, ?)",
            (ts, blob),
        )
    if debug:
        print("[STORE][USER] JSON blob inserted.")


STORES: dict[Category, Callable[[Dict[str, Any], bool], None]] = {
    "GPS": _store_gps,
    "CALENDAR": _store_calendar,
    "SCREEN_USAGE": _store_screen_usage,
    "USER": _store_user,
}

# ----------------------- public entrypoint -----------------------

def send_data(category: Category, payload: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Single entrypoint.
    - Validates payload for the given category
    - Stores it into a category-specific SQLite DB
    - Returns a simple status dict
    """
    if category not in VALIDATORS:
        if debug:
            print(f"[DEBUG] Unknown category: {category}")
        return {"status": "error", "reason": f"unknown_category:{category}"}

    if debug:
        print(f"[DEBUG] category={category} received; running validator...")

    # 1) validate
    VALIDATORS[category](payload)

    if debug:
        print(f"[DEBUG] {category} Data Validated; routing to store...")

    # 2) store
    STORES[category](payload, debug=debug)

    if debug:
        print(f"[DEBUG] {category} data stored.")

    return {"status": "ok", "category": category}
