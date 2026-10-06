import uuid
import chromadb
from typing import List, Dict, Any, Optional
from core.ollama import OllamaManager


class VectorMemoryStore:
    def __init__(self, persist_dir: str = "./chroma_db", ollama: Optional[OllamaManager] = None):
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="sysagent_history")
        self.ollama = ollama or OllamaManager()

    def add_memory(self, goal: str, plan_summary: str, result_output: str, status: str = "success") -> str:
        """
        Stores an execution record into ChromaDB with vector embeddings.
        """
        doc_text = f"Goal: {goal}\nStrategy: {plan_summary}\nExecution Result: {result_output}"
        mem_id = str(uuid.uuid4())

        embedding = self.ollama.get_embedding(doc_text)

        if embedding:
            self.collection.add(
                ids=[mem_id],
                documents=[doc_text],
                embeddings=[embedding],
                metadatas=[{"goal": goal, "status": status}]
            )
        else:
            # Fallback to Chroma default embedding if Ollama embedding is unavailable
            self.collection.add(
                ids=[mem_id],
                documents=[doc_text],
                metadatas=[{"goal": goal, "status": status}]
            )

        return mem_id

    def query_similar_memories(self, query: str, n_results: int = 3) -> List[str]:
        """
        Retrieves top-N relevant past execution memories based on semantic similarity.
        """
        if self.collection.count() == 0:
            return []

        embedding = self.ollama.get_embedding(query)

        try:
            if embedding:
                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n_results, self.collection.count())
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self.collection.count())
                )

            documents = results.get("documents", [[]])
            return documents[0] if documents else []
        except Exception as e:
            print(f"[VectorMemoryStore Error] {e}")
            return []

    def get_memory_count(self) -> int:
        return self.collection.count()