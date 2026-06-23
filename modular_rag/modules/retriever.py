import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class RetrieverModule:
    def __init__(self, k: int = 8):
        self.k = k
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model,
        )

    def retrieve(self, query: str) -> list:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        results = retriever.invoke(query)
        print(f"[Retriever] Found {len(results)} chunks for query: '{query[:40]}'")
        return results
