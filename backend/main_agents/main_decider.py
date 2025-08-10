# backend/agents/main_decider.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
from backend.stores import behavior_store as beh
from backend.agents.preference_updater import load_preferences
from backend.models.gpt4o_model import run_gpt4o

def decide(
    user_id: str,
    *,
    goals: Optional[list[dict]] = None,
    calendar: Optional[dict] = None,
    prefs: Optional[dict] = None,
    api_mode: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    recent = beh.get_recent(user_id, since_minutes=120, limit=50, debug=debug)
    learned_prefs = {"learned": load_preferences(debug=debug)}  # << auto-load
    merged_prefs  = {**(prefs or {}), **learned_prefs}

    context = {
        "recent_behaviors": recent,
        "goals": goals or [],
        "calendar": calendar or {},
        "preferences": merged_prefs,
    }

    system_prompt = """You decide the most helpful next action.
Use recent behaviors, goals, calendar, and preferences (including learned).
Allowed actions: 'nudge', 'do_nothing', 'schedule', 'other'.
Return JSON: action, reason, suggested_steps, message (optional)."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]

    if not api_mode:
        return {
            "status": "ok",
            "decision": {
                "action": "noop",
                "reason": "api_mode=False (dry-run)",
                "considerations": {"recent_count": len(recent)},
                "version": "dec_v0_dryrun",
            },
        }

    raw_output = run_gpt4o(messages, debug=debug)
    return {"status": "ok", "decision": raw_output}