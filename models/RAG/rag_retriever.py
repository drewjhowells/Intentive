from sentence_transformers import SentenceTransformer, util
import os

class RAGRetriever:
    def __init__(self, folder_path="RAG"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.docs = []
        self.embeddings = []

        for fname in os.listdir(folder_path):
            if not fname.endswith(".txt"):
                continue 
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
                for chunk in chunks:
                    if len(chunk.split(".")) <= 3 and len(chunk) <= 300:
                        self.docs.append(chunk)
                        self.embeddings.append(self.model.encode(chunk, convert_to_tensor=True))

    def retrieve(self, query, top_k=2):
        query_vec = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_vec, self.embeddings, top_k=top_k)[0]
        return [self.docs[hit['corpus_id']] for hit in hits]
