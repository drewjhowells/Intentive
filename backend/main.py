from __future__ import annotations
import os, glob, sqlite3, sys, json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


def gather_recent_payload(stores_dir: str, minutes: int, debug: bool = False) -> dict:
    """
    Minimal collector for Activity Guesser.
    - Scans all *.db in `stores_dir` (each file = category).
    - For each DB, finds tables and tries a simple time filter:
        * If a table has 'timestamp' (point event), use BETWEEN.
        * Else if it has both 'start_time' and 'end_time' (interval), use overlap.
      (Overlap = any part of the event touches the window.)
    - Returns one compact payload:
        {
          "window": {"start": ISO, "end": ISO},
          "categories": {
            "<category>": [
              {"table": "<name>", "columns": [...], "rows": [ {col: val, ...}, ... ]},
              ...
            ],
            ...
          }
        }
    Notes:
      - Times are ISO-8601 Z (UTC). Keep your DB times in ISO for string comparison.
      - If a table has none of the expected time columns, it’s skipped (by design).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=minutes)
    to_iso = lambda dt: dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    window = {"start": to_iso(since), "end": to_iso(now)}

    payload = {"window": window, "categories": {}}

    for path in glob.glob(os.path.join(stores_dir, "*.db")):
        category = os.path.splitext(os.path.basename(path))[0]  # e.g., "gps.db" -> "gps"
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            if debug:
                print(f"Processing {category} from {path}")
        except Exception:
            if debug:
                print(f"Failed to open {path}: {sys.exc_info()[1]}")
            continue  # unreadable db — ignore

        try:
            cat_items = []
            for (tname,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                # Inspect columns once (PRAGMA = SQLite table schema)
                cols = [r[1] for r in con.execute(f"PRAGMA table_info({tname})")]
                has_ts = "timestamp" in cols
                has_interval = ("start_time" in cols) and ("end_time" in cols)

                if has_ts:
                    q = f"SELECT * FROM {tname} WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
                    cur = con.execute(q, (window["start"], window["end"]))
                elif has_interval:
                    q = f"SELECT * FROM {tname} WHERE NOT (end_time < ? OR start_time > ?) ORDER BY start_time ASC"
                    cur = con.execute(q, (window["start"], window["end"]))
                else:
                    if debug:
                        print(f"Skipping {tname} in {category}: no recognized time fields")
                    continue  # no time fields we recognize

                rows = [dict(r) for r in cur.fetchall()]
                if rows:
                    cat_items.append({"table": tname, "columns": cols, "rows": rows})
                    if debug:
                        print(f"Found {len(rows)} rows in {tname} for {category}")

            if cat_items:
                payload["categories"][category] = cat_items
        finally:
            con.close()
            if debug:
                print(f"Finished processing {category}")

    return payload

def get_recent_activity(minutes: int = 60) -> dict:
    """
    Reads recent entries from backend/stores/activity_log.jsonl.
    Returns a compact payload with only those entries from the last `minutes`.
    """
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),  # go from main_agents to backend
        "stores",
        "activity_log.jsonl"
    )

    if not os.path.exists(log_path):
        return {"window": None, "activities": []}

    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=minutes)

    recent_entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                ts = datetime.strptime(entry["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if ts >= since:
                    recent_entries.append(entry)
            except Exception:
                continue  # skip malformed lines

    return {
        "window": {
            "start": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": now.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "activities": recent_entries
    }

def get_recent_calendar(minutes: int = 60) -> dict:
    """
    Placeholder for future calendar retrieval logic.
    Currently returns an empty dictionary.
    """
    return {"window": None, "events": []}  # Replace with actual calendar retrieval logic when implemented

def get_and_store_goals(goals=None, debug=False):
    """
    Placeholder for receiving user goals from the frontend.
    Currently accepts a goals list manually for testing.
    Stores them in stores/goals.json with a timestamp.

    Parameters:
        goals (list): A list of goal strings from the user.
        debug (bool): If True, prints debug output.
    """
    
    # --- Placeholder: replace with frontend data fetching later ---
    if goals is None:
        # Simulated frontend input for testing
        goals = [
            "Finish weekly project report",
            "Go for a 30-minute walk",
            "Read 10 pages of a book"
        ]
    
    # Prepare storage path
    stores_dir = os.path.join(os.path.dirname(__file__), "stores")
    os.makedirs(stores_dir, exist_ok=True)
    goals_path = os.path.join(stores_dir, "goals.json")
    
    # Load existing goals (if any)
    if os.path.exists(goals_path):
        with open(goals_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new goals with timestamp
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "goals": goals
    }
    data.append(entry)
    
    # Save back to file
    with open(goals_path, "w") as f:
        json.dump(data, f, indent=4)
    
    if debug:
        print(f"[DEBUG] Stored goals: {entry}")
        print(f"[DEBUG] Goals saved to {goals_path}")

def get_preferences():
    """
    Placeholder for future preferences retrieval logic.
    Currently returns an empty dictionary.
    """
    return {}  # Replace with actual preferences retrieval logic when implemented