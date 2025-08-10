# data_collector.py
from __future__ import annotations
from typing import Dict, Any, Literal, Callable
import sqlite3
from datetime import datetime, timezone

Category = Literal["GPS", "PHONE", "USER"]

# --- empty validators (placeholders) ---

def _validate_gps(payload: Dict[str, Any]) -> None:
    # TODO: add real checks later
    return True

def _validate_phone(payload: Dict[str, Any]) -> None:
    # TODO: add real checks later
    return

def _validate_user(payload: Dict[str, Any]) -> None:
    # TODO: add real checks later
    return

VALIDATORS: dict[Category, Callable[[Dict[str, Any]], None]] = {
    "GPS": _validate_gps,
    "PHONE": _validate_phone,
    "USER": _validate_user,
}

# --- empty stores (placeholders) ---
GPS_DB_PATH = "stores/gps.db"

def _store_gps(payload: Dict[str, Any], *, debug: bool) -> None:
    """
    Parse lat/lon from payload and store in GPS table.
    Expects payload["coords"] as a string "lat,lon" or two separate keys "lat" and "lon".
    """
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    
    # Try combined string first
    if "coords" in payload and isinstance(payload["coords"], str):
        try:
            lat_str, lon_str = payload["coords"].split(",")
            lat, lon = float(lat_str.strip()), float(lon_str.strip())
        except ValueError:
            if debug:
                print("Invalid coords format in payload:", payload["coords"])
            return
    # Try separate keys
    elif "lat" in payload and "lon" in payload:
        try:
            lat, lon = float(payload["lat"]), float(payload["lon"])
        except ValueError:
            if debug:
                print("Invalid lat/lon values in payload:", payload)
            return
    else:
        if debug:
            print("No GPS data found in payload")
        return

    with sqlite3.connect(GPS_DB_PATH) as con:
        con.execute(
            "INSERT INTO gps (ts_utc, lat, lon) VALUES (?, ?, ?)",
            (ts, lat, lon)
        )


def init_gps_store():
    """Create a simple GPS table if it doesn't exist."""
    with sqlite3.connect(GPS_DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS gps (
                ts_utc TEXT,
                lat REAL,
                lon REAL
            )
        """)

def _store_phone(payload: Dict[str, Any], *, debug: bool) -> None:
    # TODO: write to DB/queue later
    if debug:
        print("[STORE] PHONE payload accepted")

def _store_user(payload: Dict[str, Any], *, debug: bool) -> None:
    # TODO: write to DB/queue later
    if debug:
        print("[STORE] USER payload accepted")

STORES: dict[Category, Callable[[Dict[str, Any], bool], None]] = {
    "GPS": _store_gps,
    "PHONE": _store_phone,
    "USER": _store_user,
}

# --- single public entrypoint ---

def send_data(category: Category, payload: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Minimal Data Collector shell.
    - Determines category
    - Runs the (empty) validator for that category
    - Passes payload to the (empty) store function
    - Prints only when debug=True
    """
    if category not in VALIDATORS:
        if debug:
            print(f"[DEBUG] Unknown category: {category}")
        return {"status": "error", "reason": f"unknown_category:{category}"}

    if debug:
        print(f"[DEBUG] category={category} received; running validator...")

    # 1) validate (no-op for now)
    VALIDATORS[category](payload)

    if debug:
        print(f"[DEBUG] {category} Data Validated; routing to store...")

    # 2) store (no-op for now; prints only if debug)
    STORES[category](payload, debug=debug)
    if category == "GPS" and "lat" in payload and "lon" in payload:
        init_gps_store()  # Ensure GPS store is initialized
        valid_gps = _validate_gps(payload)  # Validate GPS payload
        if valid_gps:
            _store_gps(payload, debug=debug)

    if debug:
        print(f"[DEBUG] {category} data stored.")

    return {"status": "ok", "category": category}
