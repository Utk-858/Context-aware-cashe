from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for the closest matches to the query vector.
        Returns a list of dictionaries containing chunk IDs, scores, and raw content/metadata.
        """
        pass

    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Insert or update document chunks and their associated embeddings in the store."""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Removes a specific embedding by its ID."""
        pass
