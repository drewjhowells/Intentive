# backend/agents/activity_guesser.py
from __future__ import annotations
from typing import Dict, Any
import json
from backend.models.gpt5nano_model import run_gpt5nano

def guess(feature_bundle: Dict[str, Any], *, api_mode: bool = False, debug: bool = False) -> Dict[str, Any]:
    system_prompt = """You are an assistant that infers a user's most likely activity
from summarized mobile telemetry. Return JSON with keys: label, confidence, rationale, evidence.
The label is a short string (e.g., "WORK", "LEISURE", "TRAVEL"). Confidence is a float [0.0, 1.0].
Rationale is a human-readable explanation of the guess. Evidence is a JSON object with keys seen.
Example output:
{
    "label": "WORK",
    "confidence": 0.85,
    "rationale": "User has recent calendar events and screen usage indicating work-related activity.",
    "evidence": {
        "keys_seen": ["calendar", "screen_usage"]
    },
    "version": "gpt5nano_v0"
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(feature_bundle, ensure_ascii=False)},
    ]

    if not api_mode:
        result = {
            "label": "UNKNOWN",
            "confidence": 0.5,
            "rationale": "api_mode=False (dry-run)",
            "evidence": {"keys_seen": list(feature_bundle.keys())},
            "version": "gpt5nano_v0_dryrun",
        }
        if debug:
            print(f"[GUESS.DRYRUN] {result}")
        return {"status": "ok", "guess": result}
    if api_mode:
        raw_output = run_gpt5nano(messages, debug=debug)
        return {"status": "ok", "guess": raw_output}
