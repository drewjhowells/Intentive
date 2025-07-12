from models.emotional_analyzer_v1 import EmotionAnalyzer
from models.productivity_assistant_llm import ProductivityLLM
from RAG.rag_retriever import RAGRetriever
from models.memory_system import ShortTermMemory, LongTermSummary, VectorMemory, UserProfile
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
model_path = os.path.join(BASE_DIR, "models", "emotion_model.pth")
classes_path = os.path.join(BASE_DIR, "models", "emotion_classes.json")
context_path = os.path.join(BASE_DIR, "data", "full_context.txt")

loaded_analyzer = EmotionAnalyzer.load(model_path, classes_path)
retriever = RAGRetriever()


def main():
    with open(context_path, "r", encoding="utf-8") as f:
        core_context = f.read()
    agent = ProductivityLLM(context=core_context, short_term=ShortTermMemory(), summary=LongTermSummary(), vector_mem=VectorMemory(), user_profile=UserProfile())
    print("Context Loaded.")

    conversation = True
    message_count = 0

    while conversation:
        user_input = input("Please enter response (reply 'done' when complete): ")
        if user_input == "done":
            conversation = False
            break
        else:
            conversation = True
        if user_input.strip().lower() == "!!clear":
            print("Clearing all memory...")
            agent.short_term = ShortTermMemory()
            agent.vector_mem = VectorMemory()
            agent.summary = LongTermSummary() 
            with open("data\\long_term_summary.txt", "w") as f:
                f.write("")
            continue
        emotional_analysis = loaded_analyzer.predict([user_input])
        emotions_str = emotional_analysis[0]
        rag_chunks = retriever.retrieve(user_input)
        total_prompt = agent._build_prompt_llm(user_input, emotions_str, rag_chunks)
        if len(total_prompt) > 6000:  # Approximate ~6000 characters (~1500 tokens)
            print("Prompt too long. Trimming RAG content...")
            rag_chunks = rag_chunks[:1]
        print("Replying . . .")
        def summarizer_fn(text):
            return agent.generate_response(user_input=text, emotions=[], rag_chunks=[], max_tokens=200)
        reply = agent.generate_response(user_input, emotions=emotions_str, rag_chunks=rag_chunks, max_tokens=200)
        agent.short_term.add(user_input, reply)
        message_count += 1
        if message_count % 10 == 0:
            print("Updating long-term summary...")
            full_text = agent.short_term.get_formatted(include_timestamps=True)
            agent.summary.update_summary(full_text, summarizer_fn=summarizer_fn)
            print("Updated Summary:")
            print(agent.summary.get_summary())
        if any(keyword in user_input.lower() for keyword in ["goal", "struggle", "help", "plan"]):
            agent.vector_mem.add(user_input)
            print("Saving input to vector memory...")
        print("\nUser:", user_input)
        print("LLM:", reply)

if __name__ == "__main__":
    main()