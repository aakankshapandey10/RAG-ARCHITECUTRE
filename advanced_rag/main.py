import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from query_rewriter import rewrite_query
from reranker import rerank

load_dotenv(find_dotenv())

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def retrieve(query: str, k: int = 8):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def answer(question: str):
    # Step 1: rewrite the query into a better search query
    rewritten_query = rewrite_query(question)

    # Step 2: retrieve more chunks than naive RAG (8 instead of 4)
    chunks = retrieve(rewritten_query, k=8)

    # Step 3: rerank — cross-encoder scores all 8, keeps best 4
    top_chunks = rerank(question, chunks, top_k=4)

    # Step 4: build context from reranked chunks
    context = "\n\n---\n\n".join([chunk.page_content for chunk in top_chunks])

    # Step 5: send to Claude
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)

    print("\n--- ANSWER ---")
    print(response.content)
    print("\n--- SOURCES ---")
    for i, chunk in enumerate(top_chunks):
        print(f"[{i+1}] {chunk.page_content[:80]}...")


if __name__ == "__main__":
    print("Advanced RAG — ask a question about your documents.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer(question)
        print()
