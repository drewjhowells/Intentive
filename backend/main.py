from __future__ import annotations
from typing import Dict, Any, Optional, Literal

from backend.main_agents.activity_guesser import guess
from backend.main_agents.main_decider import decide
from backend.main_agents.nudger import send_nudge
from backend.main_agents.feedback_handler import handle_feedback
from backend.stores.behaviors_store import record_behavior

Category = Literal["GPS", "PHONE", "USER"]

# 1) Log system info (PHONE/GPS/etc.)
def log_system_info(user_id: str, category: str, data: Dict[str, Any], *, debug: bool = False):
    feature_bundle = {"category": category, "data": data}  
    guessed_activity = guess(feature_bundle, api_mode=False, debug=debug)
    guess_obj = guessed_activity["guess"] if "guess" in guessed_activity else guessed_activity
    record_behavior(user_id, {
        "label": guess_obj.get("label", "UNKNOWN"),
        "confidence": float(guess_obj.get("confidence", 0.5)),
        "context": feature_bundle,
        "evidence": guess_obj.get("evidence", {}),
        "version": guess_obj.get("version", "gpt5nano_v0"),
    }, debug=debug)
    return {"status": "ok", "guess": guess_obj}

# 2) Log user info (pass for now)
def log_user_info(user_id: str, data: Dict[str, Any], *, debug: bool = False):
    # Placeholder: ignore content, just ACK
    if debug: print("[USER-INFO] pass-through (not stored yet)")
    return {"status": "ok", "note": "pass"}

# 3) Get chatbox response (pass for now)
def get_chatbox_response(user_id: str, prompt: str, *, debug: bool = False):
    if debug: print("[CHATBOX] pass-through (not implemented)")
    return {"status": "ok", "response": None}

# 4) Evaluate & notify (better name than 'process_for_nudge')
def evaluate_and_notify(
    user_id: str,
    *,
    goals: Optional[list[dict]] = None,
    calendar: Optional[dict] = None,
    prefs: Optional[dict] = None,
    event_details: Optional[Dict[str, Any]] = None,
    api_mode_decide: bool = False,
    debug: bool = False,
):
    dec = decide(
        user_id,
        goals=goals,
        calendar=calendar,
        prefs=prefs,
        api_mode=api_mode_decide,
        debug=debug
    )
    decision = dec["decision"]
    nudged = send_nudge(decision, event_details=event_details, debug=debug)
    return {"status": "ok", "decision": decision, "nudger_result": nudged}

# Optional: feedback entrypoint (so FE can report user action)
def record_feedback(feedback: Literal["accept","reject","ignore"], nudger_result: Dict[str, Any], *, debug: bool = False):
    return handle_feedback(feedback=feedback, nudger_result=nudger_result, debug=debug)