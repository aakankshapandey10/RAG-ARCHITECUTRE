from corrective_rag import run_crag

if __name__ == "__main__":
    print("Corrective RAG — grades retrieved docs, falls back to web if needed.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer = run_crag(question)
            print(f"\n--- FINAL ANSWER ---\n{answer}\n")
