from pipeline import ModularPipeline

pipeline = ModularPipeline(
    use_router=True,
    use_filter=True,
    use_reranker=True,
)

if __name__ == "__main__":
    print("Modular RAG — ask a question about your documents.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer = pipeline.run(question)
            print(f"\n--- ANSWER ---\n{answer}\n")
