from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import json
from pathlib import Path
from data_schema import UserContext
from dataclasses import asdict, is_dataclass

@dataclass
class Goal:
    text: str               # e.g., “Walk 10,000 steps daily”
    created_at: datetime    # when it was set
    progress: float = 0.0   # percent complete

@dataclass
class EmotionSnapshot:
    emotion: str            # e.g., “sad”
    intensity: str          # e.g., “mild”
    timestamp: datetime     # when it was felt

@dataclass
class UserContext:
    goals: List[Goal] = field(default_factory=list)
    emotion_history: List[EmotionSnapshot] = field(default_factory=list)
    last_prompt: Optional[str] = None




CONTEXT_PATH = Path("user_context.json")

def save_context(context: UserContext):
    with open(CONTEXT_PATH, "w") as f:
        json.dump(asdict(context), f, indent=2, default=str)

def load_context() -> UserContext:
    if CONTEXT_PATH.exists():
        with open(CONTEXT_PATH) as f:
            data = json.load(f)
        # convert dict back to UserContext (with datetime parsing if needed)
        return UserContext(**data)
    return UserContext()



class SessionContext:
    def __init__(self):
        self.prompts = []
        self.current_emotion = None
        self.goal_summary = None

    def record_prompt(self, text):
        self.prompts.append(text)

    def set_emotion(self, emotion):
        self.current_emotion = emotion

    def summarize_context(self):
        return {
            "goal": self.goal_summary,
            "emotion": self.current_emotion,
            "recent_input": self.prompts[-3:]
        }
