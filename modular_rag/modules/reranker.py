from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerModule:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, question: str, chunks: list) -> list:
        if not chunks:
            return []

        pairs = [(question, chunk.page_content) for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in scored[:self.top_k]]
        print(f"[Reranker] {len(chunks)} chunks → kept top {self.top_k}")
        return top_chunks
