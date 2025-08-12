# backend/agents/feedback_handler.py
from __future__ import annotations
from typing import Dict, Any
from backend.main_agents.preference_manager import update_preferences

def handle_feedback(
    *,
    feedback: str,                 # "accept" | "reject" | "ignore"
    nudger_result: Dict[str, Any], # the dict returned by send_nudge(...)
    debug: bool = False
) -> Dict[str, Any]:
    """
    Extracts context from the nudger payload and stores a learned preference.
    """
    if nudger_result.get("status") != "sent":
        return {"status": "skipped", "reason": "nothing_sent"}

    payload = nudger_result["payload"]
    ctx = {
        "action": payload.get("action"),
        "message": payload.get("message"),
        "suggested_steps": payload.get("suggested_steps", []),
        "event_details": payload.get("event_details", {}),
        "decision": payload.get("decision_snapshot", {}),
        "timestamp": payload.get("timestamp"),
    }
    # Example policy: push raw context to updater (it will evolve later)
    return update_preferences(feedback, ctx, debug=debug)