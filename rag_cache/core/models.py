from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

@dataclass
class ResolveInput:
    """Payload required to query the cache for an existing answer."""
    query: str
    doc_ids: List[str]
    intent: str = "default"
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoreInput:
    """Payload required to save a newly generated LLM response into the cache."""
    query: str
    response: str
    doc_ids: List[str]
    intent: str = "default"
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """The normalized object serialized and stored in the KeyValueStore."""
    query: str
    response: str
    doc_ids: List[str]
    intent: str
    doc_versions: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class CacheResult:
    """The result returned to the user upon attempting to resolve cache."""
    hit: bool
    response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
