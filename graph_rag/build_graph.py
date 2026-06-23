import os
import json
import networkx as nx
from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")

EXTRACT_TEMPLATE = """Extract entities and relationships from this text.

Return a JSON object with this exact format:
{{
  "entities": ["entity1", "entity2", "entity3"],
  "relationships": [
    {{"from": "entity1", "relation": "RELATION_TYPE", "to": "entity2", "context": "brief explanation"}},
    {{"from": "entity2", "relation": "RELATION_TYPE", "to": "entity3", "context": "brief explanation"}}
  ]
}}

Rules:
- Entities: important concepts, tools, techniques, or systems mentioned
- Relations: use uppercase like IS_TYPE_OF, IMPROVES, USES, ADDS, REQUIRES, PART_OF
- Keep entities short (2-4 words max)
- Only extract what is clearly stated in the text
- Return ONLY the JSON, no other text

Text:
{text}"""


def get_llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        base_url=os.getenv("AZURE_BASE_URL"),
        api_key=os.getenv("AZURE_API_KEY"),
    )


def load_and_split():
    documents = []
    for filename in os.listdir(DATA_PATH):
        filepath = os.path.join(DATA_PATH, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} doc(s), split into {len(chunks)} chunks")
    return chunks


def extract_entities_and_relations(text: str) -> dict:
    prompt = ChatPromptTemplate.from_template(EXTRACT_TEMPLATE)
    llm = get_llm()
    result = (prompt | llm).invoke({"text": text}).content.strip()

    try:
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()
        return json.loads(result)
    except Exception:
        return {"entities": [], "relationships": []}


def build_graph(chunks):
    G = nx.DiGraph()
    node_contexts = {}

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        extracted = extract_entities_and_relations(chunk.page_content)

        for entity in extracted.get("entities", []):
            entity = entity.strip()
            if entity:
                if entity not in node_contexts:
                    node_contexts[entity] = []
                node_contexts[entity].append(chunk.page_content[:200])
                G.add_node(entity)

        for rel in extracted.get("relationships", []):
            src = rel.get("from", "").strip()
            tgt = rel.get("to", "").strip()
            relation = rel.get("relation", "RELATED_TO")
            context = rel.get("context", "")
            if src and tgt:
                G.add_node(src)
                G.add_node(tgt)
                G.add_edge(src, tgt, relation=relation, context=context)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, node_contexts


def save_graph(G, node_contexts):
    data = {
        "nodes": [
            {"id": node, "contexts": node_contexts.get(node, [])}
            for node in G.nodes()
        ],
        "edges": [
            {
                "from": u,
                "to": v,
                "relation": G[u][v].get("relation", "RELATED_TO"),
                "context": G[u][v].get("context", "")
            }
            for u, v in G.edges()
        ]
    }
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved to: {GRAPH_PATH}")


if __name__ == "__main__":
    chunks = load_and_split()
    G, node_contexts = build_graph(chunks)
    save_graph(G, node_contexts)
    print("Graph building complete.")
