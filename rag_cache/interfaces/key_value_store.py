from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class KeyValueStore(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the cached LLM response and associated metadata for a specific payload key.
        Returns None on a cache miss.
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        """Store the generated response and metadata for a given key, with an optional Time-To-Live."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a specific key from the cache (used for manual invalidation). Returns True if deleted."""
        pass
