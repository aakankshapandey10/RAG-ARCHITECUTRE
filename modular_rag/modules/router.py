import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

ROUTER_TEMPLATE = """You are a query router. Given a user question, decide where to look for the answer.

Reply with ONLY one of these two words:
- "vectordb" if the question is about RAG, AI, language models, or document retrieval
- "websearch" if the question needs real-time or external information not likely in documents

Question: {question}

Answer (one word only):"""


class RouterModule:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            base_url=os.getenv("AZURE_BASE_URL"),
            api_key=os.getenv("AZURE_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_template(ROUTER_TEMPLATE)

    def route(self, question: str) -> str:
        chain = self.prompt | self.llm
        result = chain.invoke({"question": question})
        decision = result.content.strip().lower()
        if "websearch" in decision:
            route = "websearch"
        else:
            route = "vectordb"
        print(f"[Router] '{question[:40]}...' → {route}")
        return route
