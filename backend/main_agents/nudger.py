# backend/agents/nudger.py
from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime, timezone

def send_nudge(
    decision: Dict[str, Any],
    *,
    event_details: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    action = decision.get("action", "noop")
    now_iso = datetime.now(timezone.utc).isoformat()

    if action not in ("nudge", "schedule"):
        if debug: print(f"[NUDGER] No send. Action={action} at {now_iso}")
        return {"status": "skipped", "reason": f"action={action} not sendable", "timestamp": now_iso}

    payload = {
        "timestamp": now_iso,
        "action": action,
        "message": decision.get("message", "No message provided"),
        "suggested_steps": decision.get("suggested_steps", []),
        "event_details": event_details or {},
        "decision_snapshot": decision,  # <- echo back for feedback
    }
    if debug:
        print(f"[NUDGER] Sending {action.upper()} at {now_iso}")
        print(f"         Payload: {payload}")
    return {"status": "sent", "payload": payload}