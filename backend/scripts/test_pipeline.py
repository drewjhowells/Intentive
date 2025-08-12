"""
End-to-end smoke test for Intentive:
- Seeds GPS/Calendar/Screen/User (optional)
- Runs decide()
- Mocks a nudger "send" result
- Sends feedback (accept/reject/ignore)
- Reads back recent preferences
"""

import sys
import os
from datetime import datetime, timedelta, timezone

# Make repo root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# === Imports from your codebase ===
from backend.main_agents.data_collector import send_data
from backend.main import (
    gather_recent_payload,
    get_recent_activity,
    get_recent_calendar,
    get_goals,
    get_preferences,  # the recency-aware function above
)
from backend.main_agents.activity_guesser import guess
from backend.main_agents.main_decider import decide
from backend.main_agents.feedback_handler import handle_feedback
from backend.main_agents.preference_manager import update_preferences, load_preferences

# -------- Seed Payloads --------
def iso_now(offset_minutes=0):
    """Return ISO8601 timestamp with Z, optionally offset by minutes."""
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).isoformat().replace("+00:00", "Z")

GPS_PAYLOAD = {
    "coords": "37.7749, -122.4194",
}

CALENDAR_PAYLOAD = {
    "events": [
        {
            "title": "Focus Block",
            "start_time": iso_now(+30),   # 30 min from now
            "end_time": iso_now(+90),
            "location": "Home Office",
            "description": "Deep work session",
        },
        {
            "title": "Walk",
            "start_time": iso_now(-120),  # 2 hours ago
            "end_time": iso_now(-90),
            "location": "Neighborhood",
            "description": "Break walk",
        },
    ]
}

SCREEN_PAYLOAD = {
    "sessions": [
        {
            "app_name": "YouTube",
            "package_name": "com.google.android.youtube",
            "start_time": iso_now(-180),
            "end_time": iso_now(-175),
            "duration_seconds": 300,
        },
        {
            "app_name": "Slack",
            "package_name": "com.slack",
            "start_time": iso_now(-170),
            "end_time": iso_now(-160),
            "duration_seconds": 600,
        },
    ]
}

USER_PAYLOAD = {
    "data_json": {
        "user_id": "12345",
        "preferences": {"theme": "dark", "notifications": True},
    },
}

# -------- Preferences Payload (seed) --------
PREFERENCES_PAYLOAD = {
    "events": [
        {
            "feedback": "accept",
            "context": {
                "action": "FocusBlock",
                "message": "Block 90min deep work",
                "suggested_steps": ["Silence phone", "Close Slack", "Open IDE"],
                "event_details": {"title": "Focus Block", "duration_min": 90},
                "decision": {"reason": "open time window + energy good"},
                "timestamp": iso_now(-1440),  # 24h ago
            },
        },
        {
            "feedback": "reject",
            "context": {
                "action": "WalkBreak",
                "message": "10min walk",
                "suggested_steps": ["Shoes", "Timer 10m"],
                "event_details": {"title": "Walk"},
                "decision": {"reason": "too many consecutive sits"},
                "timestamp": iso_now(-60),  # 1h ago
            },
        },
    ]
}


def seed_preferences(events, *, debug=False):
    """
    Push synthetic feedback events via update_preferences().
    Uses the same storage path as runtime (preferences.json).
    """
    for e in events:
        update_preferences(e["feedback"], e["context"], debug=debug)


# -------- Mock nudger + judge wiring --------
def mock_nudger_payload(action: str, message: str):
    """
    Minimal nudger-like payload your feedback handler expects.
    """
    return {
        "action": action,
        "message": message,
        "suggested_steps": [],
        "event_details": {},
        "decision_snapshot": {"why": "test"},
        "timestamp": iso_now(),
    }

def run_feedback_cycle(action: str, message: str, decision_feedback: str, *, debug=False):
    """
    Simulate: decide() -> nudger sends -> user feedback -> handle_feedback()
    """
    nudger_result = {
        "status": "sent",
        "payload": mock_nudger_payload(action, message),
    }
    res = handle_feedback(feedback=decision_feedback, nudger_result=nudger_result, debug=debug)
    if debug:
        print(f"[FEEDBACK] {decision_feedback} -> {res}")
    return res


def main(debug=True):
    # 1) Optional: seed device data into stores
    # Comment/uncomment as needed
    # send_data("GPS", GPS_PAYLOAD, debug=debug)
    # send_data("CALENDAR", CALENDAR_PAYLOAD, debug=debug)
    # send_data("SCREEN_USAGE", SCREEN_PAYLOAD, debug=debug)
    # send_data("USER", USER_PAYLOAD, debug=debug)

    # 2) Seed preferences directly (acts like historical feedback)
    seed_preferences(PREFERENCES_PAYLOAD["events"], debug=debug)

    # 3) Gather features → guess activity (optional but useful)
    feature_bundle = gather_recent_payload("stores", minutes=180, debug=debug)
    guess_result = guess(feature_bundle, api_mode=False, debug=debug)
    if debug:
        print(f"[GUESS] {guess_result}")

    # 4) Decide next action
    goals = get_goals()
    recent_calendar = get_recent_calendar()
    recent_activity = get_recent_activity()
    prefs_recent = get_preferences(days_back=7, debug=debug)

    action = decide(
        goals=goals,
        recent_calendar=recent_calendar,
        prefs=prefs_recent,
        recent_activity=recent_activity,
        api_mode=False,
        debug=debug,
    )
    print("[DECIDE]", action)

    # 5) Send a few mock nudges and record different feedback paths
    run_feedback_cycle("FocusBlock", "90m deep work now?", "accept", debug=debug)
    run_feedback_cycle("WalkBreak", "Quick 10m walk?", "reject", debug=debug)
    run_feedback_cycle("Hydrate", "Drink water", "ignore", debug=debug)

    # 6) Verify preferences persisted and recency filter works
    all_prefs = load_preferences(debug=debug)
    print(f"[PREFS] total={len(all_prefs)} recent(7d)={len(prefs_recent)}")


if __name__ == "__main__":
    main(debug=True)