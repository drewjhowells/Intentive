from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os
import multiprocessing
from models.memory_system import ShortTermMemory, LongTermSummary, VectorMemory, UserProfile, build_prompt_memory

import torch

num_cores = multiprocessing.cpu_count()
torch.set_num_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

class ProductivityLLM:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct", attn_implementation='eager', device=None, cpu_threads=(num_cores - 0), context=None, short_term=None, summary=None, vector_mem=None, user_profile=None):
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        torch.set_num_threads(cpu_threads)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {cpu_threads} threads on {self.device}.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # use float32 for CPU
        ).to(self.device)

        self.context = context

        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("Model Loaded.")

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
            emotions = emotions[0] if emotions else []

        prompt = self._build_prompt_llm(user_input, emotions, rag_chunks or [])

        messages = [
    {"role": "system", "content": self.context.strip()},
    {"role": "user", "content": self._build_prompt_llm(user_input, emotions, rag_chunks or [])}
]

        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.device)

        output = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.5,
            top_p=0.85,
            do_sample=True
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=False)
        if "<|assistant|>" in decoded:
            return decoded.split("<|assistant|>")[-1].strip()
        return decoded.strip()