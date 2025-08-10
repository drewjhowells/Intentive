# backend/models/gpt4o_model.py
from __future__ import annotations
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def run_gpt4o(messages: List[Dict[str, str]], *, debug: bool = False) -> str:
    """
    Minimal wrapper for GPT-4o.
    Accepts a messages list, returns raw model output (string).
    """
    if debug:
        print(f"[GPT4O] Messages: {messages}")

    # ---- LIVE CALL----
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_4O"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2
    )
    output_text = resp.choices[0].message.content
    return output_text