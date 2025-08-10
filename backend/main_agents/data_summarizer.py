# data_summarizer.py
from __future__ import annotations
from typing import Dict, Any, Literal

Category = Literal["GPS", "PHONE", "USER"]

def summarize(category: Category, payload: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    Minimal Data Summarizer shell.
    - Accepts the same {category, payload} as Data Collector
    - For now: returns a 'summary' dict that mirrors payload
    - Prints only when debug=True
    """
    if category not in ("GPS", "PHONE", "USER"):
        if debug:
            print(f"[SUM] Unknown category: {category}")
        return {"status": "error", "reason": f"unknown_category:{category}"}

    if debug:
        print(f"[SUM] category={category} received; producing summary...")

    # --- future: do real summarization per category ---
    summary = {
        "category": category,
        "summary_version": "v0",
        "data": payload,   # placeholder; later: staypoints/app sessions/etc.
    }

    if debug:
        print(f"[SUM] category={category} summary ready")

    return {"status": "ok", "summary": summary}
