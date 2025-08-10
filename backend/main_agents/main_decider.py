# backend/agents/main_decider.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
from backend.stores import behavior_store as beh
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
    context = {
        "recent_behaviors": recent,
        "goals": goals or [],
        "calendar": calendar or {},
        "preferences": prefs or {},
    }

    system_prompt = """You are an assistant that decides the most helpful next action for a user.
Base your decision on recent behaviors, goals, calendar, and preferences.
Allowed actions: 'nudge', 'do_nothing', 'schedule', 'other'.
Return JSON with keys: action, reason, suggested_steps.
"""

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
