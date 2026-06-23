import os
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


class GeneratorModule:
    def __init__(self):
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            base_url=os.getenv("AZURE_BASE_URL"),
            api_key=os.getenv("AZURE_API_KEY"),
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def generate(self, question: str, chunks: list) -> str:
        context = "\n\n---\n\n".join([chunk.page_content for chunk in chunks])
        prompt = self.prompt.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        print("[Generator] Answer generated.")
        return response.content
