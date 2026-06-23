from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"


class FilterModule:
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.model = SentenceTransformer(MODEL_NAME)

    def filter(self, question: str, chunks: list) -> list:
        if not chunks:
            return []

        question_embedding = self.model.encode(question)
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)

        scores = np.dot(chunk_embeddings, question_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(question_embedding)
        )

        filtered = [
            chunk for chunk, score in zip(chunks, scores)
            if score >= self.threshold
        ]
        print(f"[Filter] {len(chunks)} chunks → {len(filtered)} passed threshold {self.threshold}")
        return filtered
