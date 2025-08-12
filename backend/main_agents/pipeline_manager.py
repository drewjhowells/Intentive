"""
pipeline_manager_llm.py

LLM-first pipeline manager:
- Calls the LLM with a compact roll-up to produce actions.
- Sends nudges (outside this file) when instructed.
- Parses user replies via a tiny LLM call.
"""

from typing import Dict, Any, List, Callable

def run_pipeline_manager(call_llm: Callable[[str, str], Dict], daily_rollup: Dict) -> Dict:
    """
    call_llm(role, content) -> parsed JSON dict
    role: "manager" to plan actions, "parser" to parse replies
    daily_rollup: summarized flows + policy + recent_nudges
    Returns the manager action plan (JSON).
    """
    # Build prompts per the templates above (omitted for brevity)
    sys_prompt = "You are the Pipeline Manager. Output STRICT JSON only. Decide accuracy issues and user nudges. Respect policy."
    user_prompt = f"Given this daily roll-up JSON, produce actions per the schema... JSON:\n{daily_rollup}"

    # Manager call
    plan = call_llm("manager", sys_prompt + "\n\n" + user_prompt)
    return plan

def parse_user_reply(call_llm: Callable[[str, str], Dict], correlation_id: str, nudge_message: str, user_text: str) -> Dict:
    """Parse a user's free-text reply into a structured confirmation JSON."""
    sys_prompt = 'You are a parser. Output STRICT JSON only with fields: {"correlation_id":"...","confirmed":true|false,"true_label": "<optional or null>"}'
    user_prompt = f"Nudge: {nudge_message} (correlation_id={correlation_id})\nUser reply text: \"{user_text}\""
    return call_llm("parser", sys_prompt + "\n\n" + user_prompt)