# backend/models/gpt4o_model.py
from __future__ import annotations
from typing import Dict, Any, List
import json
import os

# Optional: integrate your gatekeeper (pass-through is fine if you haven't added rules)
try:
    from backend.main_agents.gatekeeper import gatekeep  # if you created it
except Exception:
    def gatekeep(text: str, **kwargs):  # noop fallback
        return text

SYSTEM_PROMPT = """You are an assistant that infers user behavior from mobile telemetry summaries.
Return STRICT JSON with keys: label, confidence, rationale, evidence.
Allowed labels: DRIVING, AT_PLACE, SCREEN_TIME, FAST_FOOD_VISIT, UNKNOWN.
- confidence is 0..1 (float)
- rationale is short (<= 2 sentences)
- evidence is a dict of salient fields you used
"""

def _build_messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
    user_prompt = {
        "task": "Infer most likely behavior.",
        "instructions": {
            "output_format": {"label": "str", "confidence": "float", "rationale": "str", "evidence": "object"},
            "allowed_labels": ["DRIVING","AT_PLACE","SCREEN_TIME","FAST_FOOD_VISIT","UNKNOWN"],
        },
        "context": context,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]

def _parse_json_safe(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # Attempt to find first/last brace as a crude rescue
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return {"label": "UNKNOWN", "confidence": 0.5, "rationale": "parse_error", "evidence": {}, "version": "gpt4o_v0"}

def infer_behavior(context: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Calls GPT-4o to infer behavior from 'context'.
    Assumes you will flip callers to api_mode=True when ready.
    """
    messages = _build_messages(context)
    # Gatekeep input (optional)
    input_text = gatekeep(json.dumps(messages, ensure_ascii=False), stage="input", debug=debug)

    # ---- LIVE CALL (fill in once you’re ready) ----
    # Example placeholder using OpenAI SDK (commented to avoid accidental spend):
    #
    # from openai import OpenAI
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # resp = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=messages,
    #     temperature=0.2,
    #     response_format={"type": "json_object"},
    # )
    # raw_text = resp.choices[0].message.content
    #
    # Until you wire the SDK, error if someone calls live mode by accident:
    raise RuntimeError("LLM not wired yet: install SDK and uncomment live call in gpt4o_model.py")

    # If wired, continue:
    # output_text = gatekeep(raw_text, stage="output", debug=debug)
    # data = _parse_json_safe(output_text)
    # data["version"] = "gpt4o_v1"
    # if debug:
    #     print(f"[GPT4O] {data}")
    # return data
