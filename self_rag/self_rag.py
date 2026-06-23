import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(find_dotenv())

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
MAX_REGENERATIONS = 2


def get_llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


def retrieve(query: str, k: int = 6) -> list:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)


# Reflection 1: Should I retrieve at all?
def should_retrieve(question: str) -> bool:
    prompt = ChatPromptTemplate.from_template(
        """Do you need external documents to answer this question accurately?
Reply ONLY "yes" or "no".

Question: {question}"""
    )
    result = (prompt | get_llm()).invoke({"question": question}).content.strip().lower()
    decision = "yes" in result
    print(f"[Self-RAG] Should retrieve? → {'YES' if decision else 'NO'}")
    return decision


# Reflection 2: Is each chunk relevant? (grades per chunk)
def grade_chunk(question: str, chunk_content: str) -> bool:
    prompt = ChatPromptTemplate.from_template(
        """Is this document relevant to answering the question?
Reply ONLY "yes" or "no".

Question: {question}
Document: {chunk}"""
    )
    result = (prompt | get_llm()).invoke({
        "question": question,
        "chunk": chunk_content
    }).content.strip().lower()
    return "yes" in result


def filter_relevant_chunks(question: str, chunks: list) -> list:
    relevant = []
    for i, chunk in enumerate(chunks):
        is_relevant = grade_chunk(question, chunk.page_content)
        status = "RELEVANT" if is_relevant else "IRRELEVANT"
        print(f"[Self-RAG] Chunk {i+1}: {status} — {chunk.page_content[:50]}...")
        if is_relevant:
            relevant.append(chunk)
    print(f"[Self-RAG] Kept {len(relevant)}/{len(chunks)} relevant chunks")
    return relevant


# Generate an answer
def generate(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join([c.page_content for c in chunks])
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
    )
    return (prompt | get_llm()).invoke({"context": context, "question": question}).content


# Reflection 3: Is the answer faithful? (no hallucination)
def is_faithful(answer: str, chunks: list) -> bool:
    context = "\n\n".join([c.page_content for c in chunks])
    prompt = ChatPromptTemplate.from_template(
        """Is this answer fully supported by the provided context?
Check if the answer makes claims NOT found in the context (hallucination).
Reply ONLY "yes" if faithful, or "no" if it contains unsupported claims.

Context: {context}
Answer: {answer}"""
    )
    result = (prompt | get_llm()).invoke({
        "context": context,
        "answer": answer
    }).content.strip().lower()
    faithful = "yes" in result
    print(f"[Self-RAG] Answer faithful to documents? → {'YES' if faithful else 'NO — regenerating'}")
    return faithful


# Reflection 4: Is the answer actually useful?
def is_useful(question: str, answer: str) -> bool:
    prompt = ChatPromptTemplate.from_template(
        """Does this answer actually address the question asked?
Reply ONLY "yes" if useful, or "no" if it fails to answer the question.

Question: {question}
Answer: {answer}"""
    )
    result = (prompt | get_llm()).invoke({
        "question": question,
        "answer": answer
    }).content.strip().lower()
    useful = "yes" in result
    print(f"[Self-RAG] Answer useful/complete? → {'YES' if useful else 'NO — regenerating'}")
    return useful


# Main Self-RAG loop
def run_self_rag(question: str) -> str:
    print(f"\n{'='*50}")
    print(f"[Self-RAG] Question: '{question}'")
    print(f"{'='*50}")

    # Reflection 1: Do I need to retrieve?
    if not should_retrieve(question):
        print("[Self-RAG] Answering from general knowledge")
        return get_llm().invoke(question).content

    # Retrieve and grade each chunk individually
    chunks = retrieve(question)
    relevant_chunks = filter_relevant_chunks(question, chunks)

    if not relevant_chunks:
        return "I don't know based on the provided documents."

    # Generate + reflect loop
    for attempt in range(1, MAX_REGENERATIONS + 1):
        print(f"[Self-RAG] Generating answer (attempt {attempt})...")
        answer = generate(question, relevant_chunks)

        # Reflection 3: Is it faithful?
        if not is_faithful(answer, relevant_chunks):
            continue

        # Reflection 4: Is it useful?
        if not is_useful(question, answer):
            continue

        print("[Self-RAG] Answer passed all reflections.")
        return answer

    print("[Self-RAG] Returning best available answer after max attempts.")
    return answer
