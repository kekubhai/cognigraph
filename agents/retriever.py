# Retriever Agent Node

from init import get_embeddings, get_chromadb_path
import chromadb

class RetrieverAgent:
    def __init__(self, chroma_path=None, embedder=None):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path or get_chromadb_path())
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.embedder = embedder or get_embeddings()

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        return results["documents"][0] if results["documents"] else []

def retriever_node(state):
    agent = RetrieverAgent()
    query = state.get("query", "")
    top_k = state.get("top_k", 5)
    docs = agent.retrieve(query, top_k)
    state["documents"] = docs
    return state
