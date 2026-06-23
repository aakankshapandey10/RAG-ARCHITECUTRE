import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


def answer(question: str):
    # Step 1: retrieve relevant chunks
    chunks = retrieve(question, k=4)

    if not chunks:
        print("No relevant documents found.")
        return

    # Step 2: build context string from chunks
    context = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])

    # Step 3: build the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    # Step 4: send to Claude
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    response = llm.invoke(prompt)

    print("\n--- ANSWER ---")
    print(response.content)
    print("\n--- SOURCES ---")
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}] {chunk.metadata.get('source', 'unknown')} — {chunk.page_content[:80]}...")


if __name__ == "__main__":
    print("Naive RAG — ask a question about your documents.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer(question)
        print()
