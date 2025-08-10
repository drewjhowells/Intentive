# activity_guesser.py
from __future__ import annotations
from typing import Dict, Any

ALLOWED_LABELS = ["DRIVING", "AT_PLACE", "SCREEN_TIME", "FAST_FOOD_VISIT", "UNKNOWN"]

def guess(feature_bundle: Dict[str, Any], *, api_mode: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Activity Guesser (LLM-ready).
    - If api_mode=False: returns a dry-run stub (no API spend).
    - If api_mode=True: delegates to backend.models.gpt4o_model.infer_behavior().
    """
    if not api_mode:
        # DRY-RUN: no API usage, just a safe stub you can log/assert against.
        result = {
            "label": "UNKNOWN",
            "confidence": 0.50,
            "rationale": "api_mode=False (dry-run)",
            "evidence": {"keys_seen": list(feature_bundle.keys())[:12]},
            "version": "gpt4o_v0_dryrun",
        }
        if debug:
            print(f"[GUESS.DRYRUN] {result}")
        return {"status": "ok", "guess": result}

    # Live path (wired but only used if you flip api_mode=True)
    from backend.models.gpt4o_model import infer_behavior
    result = infer_behavior(context=feature_bundle, debug=debug)
    # Optional sanity clamp
    if result.get("label") not in ALLOWED_LABELS:
        result["label"] = "UNKNOWN"
    c = result.get("confidence", 0.0)
    result["confidence"] = max(0.0, min(1.0, float(c)))
    return {"status": "ok", "guess": result}
