from collections import deque
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import json

class ShortTermMemory:
    def __init__(self, max_turns=6):
        self.buffer = deque(maxlen=max_turns)


    def add(self, user_input, assistant_response):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.buffer.append({
            "timestamp": timestamp,
            "user": user_input,
            "assistant": assistant_response
        })

    def get_formatted(self, include_timestamps=True):
        lines = []
        for turn in self.buffer:
            if include_timestamps:
                lines.append(f"[{turn['timestamp']}]")
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def to_list(self):
        """Optional: export the buffer as a list for logging or saving"""
        return list(self.buffer)
    

class LongTermSummary:
    def __init__(self, filepath="data\\long_term_summary.txt"):
        self.filepath = filepath
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write("")

    def update_summary(self, full_conversation, summarizer_fn):
        """Call your LLM to summarize the full_conversation and save it"""
        new_summary = summarizer_fn(full_conversation)
        with open(self.filepath, "w") as f:
            f.write(new_summary)

    def get_summary(self):
        with open(self.filepath, "r") as f:
            return f.read()


class VectorMemory:
    def __init__(self, embedder_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model)
        self.texts = []
        self.index = faiss.IndexFlatL2(384)  # for MiniLM

    def add(self, text):
        vec = self.embedder.encode([text])
        self.texts.append(text)
        self.index.add(np.array(vec).astype('float32'))

    def retrieve(self, query, top_k=2):
        if not self.texts:
            return ["(no vector memory yet)"] * top_k  # or []
        query_vec = self.embedder.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]


class UserProfile:
    def __init__(self, path="user_profile.json"):
        self.path = path
        self.data = self.load()

    def load(self):
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r") as f:
            return json.load(f)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def update(self, key, value):
        self.data[key] = value
        self.save()

    def get(self):
        return self.data


def build_prompt_memory(user_input, short_term, summary, vector_mem, user_profile):
    vector_recall = vector_mem.retrieve(user_input, top_k=2)
    
    return f"""
[User Profile]
{json.dumps(user_profile.get(), indent=2)}

[Long-term Summary]
{summary.get_summary()}

[Vector Memory Recall]
- {vector_recall[0]}
- {vector_recall[1]}

[Recent Conversation]
{short_term.get_formatted()}
"""
