import os
import openai
from models.memory_system import ShortTermMemory, LongTermSummary, VectorMemory, UserProfile, build_prompt_memory

class ProductivityLLM:
    def __init__(self, api_key=None, context=None, short_term=None, summary=None, vector_mem=None, user_profile=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.model_name = "gpt-4o-mini"
        self.context = context

        self.short_term = short_term or ShortTermMemory()
        self.summary = summary or LongTermSummary()
        self.vector_mem = vector_mem or VectorMemory()
        self.user_profile = user_profile or UserProfile()

    def _build_prompt_llm(self, user_input, emotion_list, rag_chunks):
        emotions_text = f"User emotional state: {', '.join(emotion_list)}" if emotion_list else ""
        rag_text = "\n".join(f"- {chunk}" for chunk in rag_chunks)
        memory = build_prompt_memory(
            user_input,
            self.short_term,
            self.summary,
            self.vector_mem,
            self.user_profile
        )

        return (
            f"{emotions_text}\n"
            f"{'Relevant info:' if rag_chunks else ''}\n{rag_text}\n\n"
            f"{memory}\n"
            f"[Current User Input]\nUser: {user_input.strip()}\nAssistant:"
        )

    def generate_response(self, user_input, emotions="", rag_chunks=None, max_tokens=500):
        if isinstance(emotions, list):
            emotions = emotions[0] if emotions else ""

        prompt = self._build_prompt_llm(user_input, emotions, rag_chunks or [])

        # Send to OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.context.strip()},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.85,
        )
        return response.choices[0].message['content'].strip()
