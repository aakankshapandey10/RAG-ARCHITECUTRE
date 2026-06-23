import os
import json
import networkx as nx
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")

ANSWER_TEMPLATE = """You are a helpful assistant. Use the knowledge graph context below to answer the question.
The context contains entities, their relationships, and relevant text passages extracted from documents.

Knowledge Graph Context:
{context}

Question: {question}

Answer based only on the provided context. If the answer cannot be found, say "I don't know based on the knowledge graph."

Answer:"""


def load_graph() -> tuple:
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(
            f"Knowledge graph not found at {GRAPH_PATH}. Run build_graph.py first."
        )

    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    node_contexts = {}

    for node in data["nodes"]:
        G.add_node(node["id"])
        node_contexts[node["id"]] = node.get("contexts", [])

    for edge in data["edges"]:
        G.add_edge(
            edge["from"],
            edge["to"],
            relation=edge.get("relation", "RELATED_TO"),
            context=edge.get("context", ""),
        )

    return G, node_contexts


def find_relevant_nodes(query: str, G: nx.DiGraph, node_contexts: dict, top_k: int = 5) -> list:
    query_words = set(query.lower().split())

    scored = []
    for node in G.nodes():
        node_words = set(node.lower().split())
        name_score = len(query_words & node_words) * 3

        context_score = 0
        for ctx in node_contexts.get(node, []):
            ctx_lower = ctx.lower()
            for word in query_words:
                if len(word) > 3 and word in ctx_lower:
                    context_score += 1

        total = name_score + context_score
        if total > 0:
            scored.append((node, total))

    scored.sort(key=lambda x: -x[1])
    return [node for node, _ in scored[:top_k]]


def build_subgraph_context(nodes: list, G: nx.DiGraph, node_contexts: dict) -> str:
    if not nodes:
        return "No relevant entities found."

    expanded = set(nodes)
    for node in nodes:
        for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
            expanded.add(neighbor)

    lines = ["=== Entities ==="]
    for node in nodes:
        contexts = node_contexts.get(node, [])
        if contexts:
            lines.append(f"\n[{node}]")
            lines.append(contexts[0])

    lines.append("\n=== Relationships ===")
    seen = set()
    for u, v, data in G.edges(data=True):
        if u in expanded and v in expanded:
            key = (u, v)
            if key not in seen:
                seen.add(key)
                rel = data.get("relation", "RELATED_TO")
                ctx = data.get("context", "")
                line = f"{u} --[{rel}]--> {v}"
                if ctx:
                    line += f"  ({ctx})"
                lines.append(line)

    return "\n".join(lines)


def answer(question: str):
    G, node_contexts = load_graph()
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    relevant_nodes = find_relevant_nodes(question, G, node_contexts)
    if not relevant_nodes:
        print("No relevant entities found in the knowledge graph.")
        return

    print(f"Relevant entities: {relevant_nodes}")
    context = build_subgraph_context(relevant_nodes, G, node_contexts)

    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    response = (prompt | llm).invoke({"context": context, "question": question})

    print("\n--- ANSWER ---")
    print(response.content)
    print("\n--- RELEVANT ENTITIES ---")
    for node in relevant_nodes:
        print(f"  • {node}")


if __name__ == "__main__":
    print("Graph RAG — answers questions using a knowledge graph.")
    print("Run build_graph.py first to build the graph from your documents.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            answer(question)
        print()
