from agent import run_agent

if __name__ == "__main__":
    print("Agentic RAG — the agent decides how to find your answer.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer = run_agent(question)
            print(f"\n--- FINAL ANSWER ---\n{answer}\n")
