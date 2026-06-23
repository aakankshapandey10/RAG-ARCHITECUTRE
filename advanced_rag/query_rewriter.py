import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

REWRITE_TEMPLATE = """You are a search query optimizer.
Your job is to rewrite the user's question into a better search query
that will find the most relevant information in a document database.

Rules:
- Keep it concise (under 20 words)
- Focus on key concepts and terms
- Remove filler words
- Return ONLY the rewritten query, nothing else

Original question: {question}

Rewritten query:"""


def rewrite_query(question: str) -> str:
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(REWRITE_TEMPLATE)
    chain = prompt | llm
    result = chain.invoke({"question": question})
    rewritten = result.content.strip()
    print(f"Original query : {question}")
    print(f"Rewritten query: {rewritten}")
    return rewritten
