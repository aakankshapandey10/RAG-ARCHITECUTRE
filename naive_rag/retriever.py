import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_retriever(k=4):
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


def retrieve(question: str, k=4):
    retriever = get_retriever(k)
    results = retriever.invoke(question)
    return results
