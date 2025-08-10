# backend/stores/behavior_store.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone

# DEV-ONLY memory store (swap for DB later)
_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

def record_behavior(user_id: str, behavior: Dict[str, Any], *, debug: bool = False) -> Dict[str, Any]:
    """
    behavior expects: {label, confidence, context?, evidence?, ts? (ISO8601), version?}
    """
    evt = {
        "ts": behavior.get("ts") or datetime.now(timezone.utc).isoformat(),
        "label": behavior.get("label", "UNKNOWN"),
        "confidence": float(behavior.get("confidence", 0.5)),
        "context": behavior.get("context", {}),
        "evidence": behavior.get("evidence", {}),
        "version": behavior.get("version", "beh_v0"),
    }
    bucket = _MEMORY.setdefault(user_id, [])
    bucket.append(evt)
    if debug:
        print(f"[BEHAV_STORE] user={user_id} label={evt['label']} p={evt['confidence']:.2f}")
    return {"status": "ok", "id": f"beh_{len(bucket)}"}

def _parse_iso(ts: str) -> datetime:
    # tolerate trailing 'Z'
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def get_recent(user_id: str, *, since_minutes: int = 120, limit: int = 100, debug: bool = False) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    events = _MEMORY.get(user_id, [])
    recent = [e for e in events if _parse_iso(e["ts"]) >= cutoff]
    recent.sort(key=lambda e: _parse_iso(e["ts"]), reverse=True)
    out = recent[:limit]
    if debug:
        print(f"[BEHAV_STORE] user={user_id} recent_count={len(out)} (since {since_minutes}m)")
    return out

def clear(user_id: Optional[str] = None) -> None:
    if user_id is None:
        _MEMORY.clear()
    else:
        _MEMORY.pop(user_id, None)
