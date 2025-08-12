"""
policy_orchestrator.py

Authoritative guardrails live here. The LLM manager is stateless and untrusted.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Policy:
    daily_nudge_budget: int = 5
    nudge_cooldown_min: int = 120
    quiet_hours: tuple[int, int] = (22, 7)  # 10pm–7am local
    opt_out: bool = False

class NudgeState:
    """Tracks recent nudges; in prod, back this with a tiny DB."""
    def __init__(self):
        self.sent = []  # list[(ts, correlation_id)]

    def count_today(self, now: datetime) -> int:
        start = datetime(now.year, now.month, now.day)
        return sum(1 for ts, _ in self.sent if ts >= start)

    def last_ts(self) -> datetime | None:
        return max((ts for ts, _ in self.sent), default=None)

def can_nudge(policy: Policy, state: NudgeState, now: datetime) -> bool:
    """Authoritative gate."""
    if policy.opt_out:
        return False
    h = now.hour
    if policy.quiet_hours[0] <= h or h < policy.quiet_hours[1]:
        return False
    if state.count_today(now) >= policy.daily_nudge_budget:
        return False
    last = state.last_ts()
    if last and now - last < timedelta(minutes=policy.nudge_cooldown_min):
        return False
    return True

def run_manager_with_policy(call_llm, rollup_json, policy: Policy, state: NudgeState, now: datetime):
    """
    1) Pre-check: if nudging is disallowed, tell the LLM to avoid nudges.
    2) Call LLM.
    3) Post-filter: drop any 'ask_user_nudge' actions that violate policy anyway.
    """
    preflag = {"nudges_allowed": can_nudge(policy, state, now)}
    rollup_json["policy_runtime_gate"] = preflag

    plan = call_llm("manager", rollup_json)  # returns JSON actions

    filtered = []
    for a in plan.get("actions", []):
        if a.get("type") == "ask_user_nudge":
            if not preflag["nudges_allowed"]:
                continue  # hard block
            # optional: de-dup by flow_id/correlation_id here
            state.sent.append((now, a["correlation_id"]))
        filtered.append(a)
    plan["actions"] = filtered
    return plan