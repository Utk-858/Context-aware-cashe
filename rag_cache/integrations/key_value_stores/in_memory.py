from typing import Dict, Any, Optional, Callable
from collections import OrderedDict
from rag_cache.interfaces.key_value_store import KeyValueStore

class InMemoryKeyValueStore(KeyValueStore):
    """
    An in-memory Key-Value store with LRU (Least Recently Used) eviction schema.
    Uses OrderedDict to tightly bound memory growth for safe continuous deployment.
    """
    def __init__(self, max_entries: int = 10000, on_evict: Optional[Callable[[str], None]] = None):
        self.max_entries = max_entries
        self.on_evict = on_evict
        # OrderedDict maintains insertion/access order
        self.storage: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Returns value and marks the item as recently used."""
        if key in self.storage:
            # Moving the accessed key to the right endpoint
            self.storage.move_to_end(key)
            return self.storage[key]
        return None

    def set(self, key: str, value: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        """Sets value, marks as recent, and evicts oldest items if full."""
        # Note: In-memory TTL not implemented for basic version
        self.storage[key] = value
        self.storage.move_to_end(key)
        
        # O(1) Eviction of the oldest unread item
        if len(self.storage) > self.max_entries:
            evicted_key, _ = self.storage.popitem(last=False)
            
            # Decoupled trigger (dependency injection) to delete from Vector Database
            if self.on_evict:
                self.on_evict(evicted_key)

    def delete(self, key: str) -> bool:
        """Deletes a key from the cache."""
        if key in self.storage:
            del self.storage[key]
            return True
        return False
