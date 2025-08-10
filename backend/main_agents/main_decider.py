# backend/agents/main_decider.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
from backend.models.gpt4o_model import run_gpt4o

def decide(
    goals: Optional[list[dict]] = None,
    recent_calendar: Optional[dict] = None,
    prefs: Optional[dict] = None,
    recent_activity: Optional[dict] = None,
    api_mode: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    context = {
        "goals": goals or [],
        "calendar": recent_calendar or {},
        "preferences": prefs,
        "recent_activity": recent_activity,
    }
    if debug:
        print("Deciding based on context:")
        print("Goals:", context["goals"])
        print("Calendar:", context["calendar"])
        print("Preferences:", context["preferences"])
        print("Recent Activity:", context["recent_activity"])

    system_prompt = """You decide the most helpful next action for user.
Use goals, calendar, preferences, andrecent behaviors.
Allowed actions: 'nudge' (aka send a notifcation), 'do_nothing', 'schedule' (aka some event in the user's calendar), 'other'.
You may return muliple actions, but only one of them can be 'nudge'.
If you return 'nudge', you must also return a reason and a message.
If you return 'schedule', you must also return a reason and event details.
If you return 'other', you must also return a reason and a message.
If you return 'do_nothing', you must also return a reason.
If you receive no information or insufficient information, return 'Not enough information' as the reason and 'do_nothing' as the action.
Return JSON: action, reason, suggested_steps, message (as needed)."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]

    if not api_mode:
        if debug:
            print("Running in dry-run mode, no actual decision will be made.")
        return {
            "status": "ok",
            "decision": {
                "action": "noop",
                "reason": "api_mode=False (dry-run)",
                "considerations": "Dry run mode, no actual decision made.",
                "version": "dec_v0_dryrun",
            },
        }
    if debug:
        print("Running in API mode, making a decision...")
    raw_output = run_gpt4o(messages, debug=debug)
    return {"status": "ok", "decision": raw_output}