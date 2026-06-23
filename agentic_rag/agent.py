import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(find_dotenv())

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

MAX_ATTEMPTS = 3


def get_llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )


# --- Agent Decision 1: Does this question need retrieval? ---
def needs_retrieval(question: str) -> bool:
    prompt = ChatPromptTemplate.from_template("""Do you need to look up information to answer this question,
or can you answer it directly from general knowledge?

Question: {question}

Reply with ONLY "yes" if you need to look up information, or "no" if you can answer directly.""")

    chain = prompt | get_llm()
    result = chain.invoke({"question": question}).content.strip().lower()
    decision = "yes" in result
    print(f"[Agent] Need retrieval? → {'YES' if decision else 'NO'}")
    return decision


# --- Agent Decision 2: What query should I search? ---
def generate_search_query(question: str, attempt: int, previous_query: str = "") -> str:
    if attempt == 1:
        template = """Convert this question into a short, focused search query (under 10 words).
Return ONLY the query, nothing else.

Question: {question}
Search query:"""
        prompt = ChatPromptTemplate.from_template(template)
        result = prompt | get_llm()
        query = result.invoke({"question": question}).content.strip()
    else:
        template = """Your previous search query "{previous_query}" did not return good results.
Generate a DIFFERENT search query for this question. Try different keywords.
Return ONLY the query, nothing else.

Question: {question}
New search query:"""
        prompt = ChatPromptTemplate.from_template(template)
        result = prompt | get_llm()
        query = result.invoke({"question": question, "previous_query": previous_query}).content.strip()

    print(f"[Agent] Search query (attempt {attempt}): '{query}'")
    return query


# --- Retrieval ---
def retrieve(query: str, k: int = 5) -> list:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(query)
    print(f"[Agent] Retrieved {len(chunks)} chunks")
    return chunks


# --- Agent Decision 3: Are these chunks good enough? ---
def grade_chunks(question: str, chunks: list) -> bool:
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    template = """You are grading whether retrieved documents are relevant to answer a question.

Question: {question}

Retrieved documents:
{context}

Are these documents relevant and sufficient to answer the question?
Reply ONLY with "yes" or "no"."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm()
    result = chain.invoke({"question": question, "context": context}).content.strip().lower()
    is_good = "yes" in result
    print(f"[Agent] Chunks relevant? → {'YES' if is_good else 'NO, retrying...'}")
    return is_good


# --- Generate Answer ---
def generate_answer(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])
    template = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | get_llm()
    answer = chain.invoke({"question": question, "context": context}).content
    return answer


# --- Main Agent Loop ---
def run_agent(question: str) -> str:
    print(f"\n{'='*50}")
    print(f"[Agent] Starting for: '{question}'")
    print(f"{'='*50}")

    # Decision 1: Do I need to retrieve?
    if not needs_retrieval(question):
        print("[Agent] Answering from general knowledge (no retrieval needed)")
        llm = get_llm()
        return llm.invoke(question).content

    # Decision 2 & 3: Search loop — try up to MAX_ATTEMPTS times
    previous_query = ""
    best_chunks = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        query = generate_search_query(question, attempt, previous_query)
        chunks = retrieve(query)

        if grade_chunks(question, chunks):
            best_chunks = chunks
            break

        previous_query = query
        if attempt == MAX_ATTEMPTS:
            print(f"[Agent] Reached max attempts ({MAX_ATTEMPTS}), using best available chunks")
            best_chunks = chunks

    # Generate final answer
    print("[Agent] Generating final answer...")
    answer = generate_answer(question, best_chunks)
    return answer
