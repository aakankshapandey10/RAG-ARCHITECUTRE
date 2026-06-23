from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(question: str, chunks: list, top_k: int = 4) -> list:
    model = get_model()

    pairs = [(question, chunk.page_content) for chunk in chunks]
    scores = model.predict(pairs)

    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    print(f"Reranked {len(chunks)} chunks → kept top {top_k}")
    return top_chunks
