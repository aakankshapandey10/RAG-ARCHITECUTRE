from self_rag import run_self_rag

if __name__ == "__main__":
    print("Self-RAG — reflects on its own retrieval and answers.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer = run_self_rag(question)
            print(f"\n--- FINAL ANSWER ---\n{answer}\n")
