import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
    print(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at: {CHROMA_PATH}")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)
    print("Ingestion complete.")
