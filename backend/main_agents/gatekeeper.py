# gatekeeper.py
# A simple gatekeeper for LLM input/output text.
# It applies rules to modify text and logs the process if debugging is enabled.
from __future__ import annotations
from typing import Callable, Iterable, Literal, Optional
import datetime as _dt

Stage = Literal["input", "output"]

def gatekeep(
    text: str,
    *,
    stage: Stage = "input",
    debug: bool = False,
    rules: Optional[Iterable[Callable[[str, Stage], Optional[str]]]] = None,
) -> str:
    """
    Single-call Gatekeeper for LLM I/O (input or output).
    - For now: returns text unchanged unless a rule modifies it.
    - If debug=True, logs a one-line PASS with timestamp, stage, and length.
    - `rules`: Optional list of callables: (text, stage) -> Optional[str]
        - return None = no change
        - return str  = modified text

    Example:
        safe_prompt = gatekeep(user_prompt, stage="input", debug=True)
        safe_reply  = gatekeep(llm_reply,   stage="output", debug=True)
    """
    # Apply any rules in order (each can transform the text)
    if rules:
        for rule in rules:
            maybe_text = rule(text, stage)
            if maybe_text is not None:
                text = maybe_text

    if debug:
        ts = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        print(f"[{ts}] GATEKEEPER PASS stage={stage} len={len(text)}")

    return text
