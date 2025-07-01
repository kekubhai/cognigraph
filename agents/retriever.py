# Retriever Agent Node

import chromadb
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings

class RetrieverAgent:
    def __init__(self, chroma_path: str, model_name: str):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        # Return the text chunks
        return results["documents"][0] if results["documents"] else []
