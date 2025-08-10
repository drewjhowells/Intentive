# Run: python -m scripts.dev_playground
from __future__ import annotations
import json
from backend.backend_main import (
    log_system_info, log_user_info, get_chatbox_response,
    evaluate_and_notify, record_feedback
)

STATE = {
    "last_decision": None,
    "last_nudged": None,
    "user_id": "user_123"
}

def step1_log_system():
    payload = {"lat": 43.49, "lon": -112.03, "speed": 0.4, "package": "com.instagram.android", "event": "start"}
    print("Logging system info (default payload).")
    print(log_system_info(STATE["user_id"], payload, debug=True))

def step2_log_user():
    data = {"note": "prefers later snacks"}
    print("Logging user info (pass).")
    print(log_user_info(STATE["user_id"], data, debug=True))

def step3_chatbox():
    print("Chatbox (pass).")
    print(get_chatbox_response(STATE["user_id"], "hello world", debug=True))

def step4_decide_and_nudge():
    goals = [{"id":"g1","title":"Eat healthier","tags":["nutrition"]}]
    calendar = {"free": []}
    prefs = {"quiet_hours": ["22:00","07:00"]}
    print("Evaluating decision (dry-run) and nudging (stub).")
    res = evaluate_and_notify(
        STATE["user_id"],
        goals=goals, calendar=calendar, prefs=prefs,
        event_details={"time":"2025-08-12T15:00:00Z"},
        api_mode_decide=False, debug=True
    )
    STATE["last_decision"] = res["decision"]
    STATE["last_nudged"] = res["nudger_result"]
    print(res)

def step5_feedback():
    if not STATE["last_nudged"]:
        print("No nudger_result in state. Run step 4 first.")
        return
    fb = record_feedback("reject", STATE["last_nudged"], debug=True)
    print(fb)

def step6_decide_again():
    res = evaluate_and_notify(
        STATE["user_id"],
        goals=None, calendar=None, prefs=None,
        event_details=None,
        api_mode_decide=False, debug=True
    )
    print(res)

MENU = """
Choose a step:
  1) Log system info
  2) Log user info (pass)
  3) Get chatbox response (pass)
  4) Decide & Nudge (dry-run)
  5) Send feedback (reject) → update prefs
  6) Decide again (after feedback)
  q) Quit
> """

def main():
    while True:
        choice = input(MENU).strip().lower()
        if choice == "1": step1_log_system()
        elif choice == "2": step2_log_user()
        elif choice == "3": step3_chatbox()
        elif choice == "4": step4_decide_and_nudge()
        elif choice == "5": step5_feedback()
        elif choice == "6": step6_decide_again()
        elif choice in ("q", "quit", "exit"): break
        else: print("Unknown option.")

if __name__ == "__main__":
    main()