Backend README

How to Use

Main entrypoint: backend_main.py
Import and call functions directly from the front end.

Example:

from backend.backend_main import log_system_info, evaluate_and_notify

# Log system info (e.g., GPS data)
log_system_info("user_123", {"lat": 43.49, "lon": -112.03}, debug=True)

# Decide if a nudge should be sent
evaluate_and_notify(
    "user_123",
    goals=[...],
    prefs={...},
    api_mode=False,  # Use True in prod to call real LLM
    debug=True
)

Modes:
	•	debug=True → Prints internal steps for visibility (safe in dev; no logic changes).
	•	api_mode=False → Uses stubbed LLM responses (no API usage or cost).
Set to True only in production to call real models.

⸻

Supplementary Info

Process Flow
	1.	log_system_info
	•	Front end sends system/context data (GPS, phone usage, etc.).
	•	Data Collector routes to correct handler.
	•	Activity Guesser (LLM or stub) labels possible behaviors and stores them.
	2.	log_user_info
	•	Accepts user profile or preference updates (stub for now).
	3.	get_chatbox_response
	•	Placeholder for real-time chat interaction with the LLM.
	4.	evaluate_and_notify
	•	Pulls latest behaviors, goals, and preferences.
	•	Main Decider uses context to determine if a nudge or action is relevant.
	•	Passes decision to Nudger.
	5.	record_feedback
	•	FE sends user feedback on past nudges.
	•	Updates stored preferences and behavior handling for future decisions.

⸻

File Roles
	•	main_agents/
	•	activity_guesser.py – Uses GPT-5-nano (or stub) to identify behaviors.
	•	main_decider.py – Uses GPT-4o (or stub) to choose actions.
	•	nudger.py – Stub for delivering chosen nudges/actions.
	•	feedback_handler.py – Updates preferences from user feedback.
	•	models/
	•	gpt4o_model.py & gpt5nano_model.py – Model wiring and prompt execution.
	•	stores/behavior_store.py
	•	Stub for saving detected behaviors.
	•	scripts/dev_playground.py
	•	Interactive CLI for testing each step or the whole pipeline.