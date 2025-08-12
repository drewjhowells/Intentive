"""
Preference Updater Agent
- Takes feedback + nudge context
- Decides what to change in future decision-making
- Saves to in-memory store (stub; replace with DB later)
"""

from typing import Dict, Any, List
import json
import os

# Simple in-memory store (stub)
PREFERENCES: List[Dict[str, Any]] = []
PREFS_FILE = "preferences.json"

def update_preferences(feedback: str, context: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Save a preference change based on user feedback.
    - feedback: e.g., "accept", "reject", "ignore"
    - context: dict containing nudge details (goal, time, location, etc.)
    """
    entry = {
        "feedback": feedback,
        "context": context
    }
    PREFERENCES.append(entry)

    if debug:
        print(f"[PREF-UPDATER] Stored preference: {entry}")

    # Save to JSON (stubbed persistence)
    try:
        with open(PREFS_FILE, "w", encoding="utf-8") as f:
            json.dump(PREFERENCES, f, indent=2)
        if debug:
            print(f"[PREF-UPDATER] Preferences saved to {PREFS_FILE}")
    except Exception as e:
        if debug:
            print(f"[PREF-UPDATER] Error saving preferences: {e}")

    return {"status": "ok", "stored": entry}

def load_preferences(*, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Load preferences from the JSON file.
    """
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                prefs = json.load(f)
            if debug:
                print(f"[PREF-UPDATER] Loaded {len(prefs)} preferences")
            return prefs
        except Exception as e:
            if debug:
                print(f"[PREF-UPDATER] Error loading preferences: {e}")
            return []
    else:
        if debug:
            print("[PREF-UPDATER] No preference file found")
        return []
