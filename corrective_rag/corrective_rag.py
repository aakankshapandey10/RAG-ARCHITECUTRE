import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from duckduckgo_search import DDGS

load_dotenv(find_dotenv())

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


# --- Step 1: Retrieve from vector DB ---
def retrieve(question: str, k: int = 6) -> list:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    chunks = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    print(f"[CRAG] Retrieved {len(chunks)} chunks from vector DB")
    return chunks


# --- Step 2: Grade each chunk, return overall score ---
def grade_chunks(question: str, chunks: list) -> tuple:
    prompt = ChatPromptTemplate.from_template(
        """Is this document relevant to answering the question?
Reply ONLY "yes" or "no".

Question: {question}
Document: {chunk}"""
    )
    llm = get_llm()
    relevant_count = 0

    for i, chunk in enumerate(chunks):
        result = (prompt | llm).invoke({
            "question": question,
            "chunk": chunk.page_content
        }).content.strip().lower()
        is_relevant = "yes" in result
        status = "RELEVANT" if is_relevant else "IRRELEVANT"
        print(f"[CRAG] Chunk {i+1}: {status}")
        if is_relevant:
            relevant_count += 1

    score = relevant_count / len(chunks) if chunks else 0
    print(f"[CRAG] Relevance score: {relevant_count}/{len(chunks)} = {score:.0%}")

    if score >= 0.6:
        decision = "high"
    elif score >= 0.3:
        decision = "medium"
    else:
        decision = "low"

    print(f"[CRAG] Decision: {decision.upper()} → ", end="")
    if decision == "high":
        print("use documents only")
    elif decision == "medium":
        print("use documents + web search")
    else:
        print("ignore documents, use web search only")

    relevant_chunks = [c for c, r in zip(chunks, [True if grade_chunk_score(question, c, llm) else False for c in chunks]) if r]
    return decision, chunks


def grade_chunk_score(question, chunk, llm) -> bool:
    prompt = ChatPromptTemplate.from_template(
        """Is this document relevant to answering the question?
Reply ONLY "yes" or "no".

Question: {question}
Document: {chunk}"""
    )
    result = (prompt | llm).invoke({
        "question": question,
        "chunk": chunk.page_content
    }).content.strip().lower()
    return "yes" in result


# --- Step 3: Web search fallback ---
def web_search(question: str, max_results: int = 4) -> str:
    print(f"[CRAG] Searching web for: '{question}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=max_results))
        if not results:
            return "No web results found."
        combined = "\n\n".join([
            f"Source: {r.get('href', 'unknown')}\n{r.get('body', '')}"
            for r in results
        ])
        print(f"[CRAG] Got {len(results)} web results")
        return combined
    except Exception as e:
        print(f"[CRAG] Web search failed: {e}")
        return "Web search unavailable."


# --- Step 4: Generate answer ---
def generate(question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using the context below.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""
    )
    return (prompt | get_llm()).invoke({
        "context": context,
        "question": question
    }).content


# --- Main CRAG pipeline ---
def run_crag(question: str) -> str:
    print(f"\n{'='*50}")
    print(f"[CRAG] Question: '{question}'")
    print(f"{'='*50}")

    # Step 1: Retrieve
    chunks = retrieve(question)

    # Step 2: Grade
    decision, _ = grade_chunks(question, chunks)

    # Step 3: Build context based on decision
    if decision == "high":
        context = "\n\n---\n\n".join([c.page_content for c in chunks])

    elif decision == "medium":
        doc_context = "\n\n---\n\n".join([c.page_content for c in chunks])
        web_context = web_search(question)
        context = f"FROM DOCUMENTS:\n{doc_context}\n\nFROM WEB:\n{web_context}"

    else:
        context = web_search(question)

    # Step 4: Generate
    print("[CRAG] Generating answer...")
    return generate(question, context)
