from modules.router import RouterModule
from modules.retriever import RetrieverModule
from modules.filter import FilterModule
from modules.reranker import RerankerModule
from modules.generator import GeneratorModule


class ModularPipeline:
    def __init__(self, use_router=True, use_filter=True, use_reranker=True):
        self.use_router = use_router
        self.use_filter = use_filter
        self.use_reranker = use_reranker

        self.router = RouterModule() if use_router else None
        self.retriever = RetrieverModule(k=8)
        self.filter = FilterModule(threshold=0.3) if use_filter else None
        self.reranker = RerankerModule(top_k=4) if use_reranker else None
        self.generator = GeneratorModule()

    def run(self, question: str) -> str:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")

        # Step 1: Route the question
        if self.use_router:
            route = self.router.route(question)
            if route == "websearch":
                return "This question requires a web search. Web search is not connected in this demo."

        # Step 2: Retrieve chunks
        chunks = self.retriever.retrieve(question)

        # Step 3: Filter low-quality chunks
        if self.use_filter and self.filter:
            chunks = self.filter.filter(question, chunks)

        if not chunks:
            return "No relevant documents found after filtering."

        # Step 4: Rerank
        if self.use_reranker and self.reranker:
            chunks = self.reranker.rerank(question, chunks)

        # Step 5: Generate answer
        answer = self.generator.generate(question, chunks)
        return answer
