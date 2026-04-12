import hashlib
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import asdict

from rag_cache.interfaces.embedding import Embedder
from rag_cache.interfaces.vector_store import VectorStore
from rag_cache.interfaces.key_value_store import KeyValueStore
from rag_cache.interfaces.intent import IntentClassifier
from rag_cache.core.decision_engine import DecisionEngine
from rag_cache.core.metrics import MetricsTracker
from rag_cache.core.models import ResolveInput, StoreInput, CacheResult, CacheEntry

class RetrievalCache:
    """
    L1 Cache Layer.
    A purely exact-match Key-Value cache mapping identical user queries 
    directly to retrieved document IDs, bypassing Document VectorDB lookups.
    """
    def __init__(self, kv_store: KeyValueStore):
        self.kv_store = kv_store
        self.metrics = MetricsTracker()

    def _generate_cache_key(self, query: str) -> str:
        return f"retrieval:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"

    def resolve(self, query: str) -> Optional[List[str]]:
        """Fetch previously retrieved Document IDs for this exact string."""
        key = self._generate_cache_key(query)
        data = self.kv_store.get(key)
        
        if data and "doc_ids" in data:
            self.metrics.record_hit()
            return data["doc_ids"]
            
        self.metrics.record_miss()
        return None

    def store(self, query: str, doc_ids: List[str]) -> None:
        """Store the Document IDs associated with this query string."""
        key = self._generate_cache_key(query)
        self.kv_store.set(key, {"doc_ids": doc_ids})
        
    def get_stats(self) -> Dict[str, Any]:
        return self.metrics.get_stats()


class GenerationCache:
    """
    L2 Cache Layer.
    A context-aware, semantic caching system orchestrator.
    Maps a semantic embedding to an LLM response, but strictly requires 
    DecisionEngine checks (Context stability, Intent) before hitting.
    """
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore, 
        kv_store: KeyValueStore,    
        intent_classifier: IntentClassifier,
        decision_engine: DecisionEngine,
        bypass_intents: Optional[List[str]] = None,
        debug_mode: bool = False
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.kv_store = kv_store
        self.intent_classifier = intent_classifier
        self.decision_engine = decision_engine
        
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.decision_engine.config.debug_mode = True
        
        self.bypass_intents = bypass_intents or ["action"]
        self.metrics = MetricsTracker()

    def get_stats(self) -> Dict[str, Any]:
        """Exposes metrics for observability/dashboards."""
        return self.metrics.get_stats()

    def _generate_cache_key(self, query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def resolve(self, resolve_input: ResolveInput) -> Tuple[CacheResult, str]:
        if self.debug_mode:
            print(f"\n[DEBUG L2] Resolving semantic cache for: '{resolve_input.query}'")
            
        # Step 1: Classify Intent
        if resolve_input.intent == "default":
            resolve_input.intent = self.intent_classifier.classify(resolve_input.query)
            
        if resolve_input.intent in self.bypass_intents:
            self.metrics.record_miss()
            return CacheResult(hit=False), "Miss: Intent configured for cache bypass."

        # Step 2: Generate Embedding
        query_vector = self.embedder.embed_query(resolve_input.query)

        # Step 3: Search Semantic Vector Store
        search_results = self.vector_store.search(query_vector, top_k=5)
        
        if not search_results:
            reason = "Miss: No semantically similar queries found in VectorStore."
            if self.debug_mode:
                print(f"[DEBUG L2] Vector search returned empty -> {reason}")
                
            self.metrics.record_miss()
            return CacheResult(hit=False), reason

        # Step 4: Fetch Candidate Payloads
        candidates_with_scores = []
        for result in search_results:
            score = result.get("score", 0.0)
            cache_key = result.get("id")
            if not cache_key: continue
                
            cached_data = self.kv_store.get(cache_key)
            if cached_data:
                try:
                    entry = CacheEntry(**cached_data)
                    candidates_with_scores.append((entry, score))
                except Exception:
                    pass

        # Step 5: Filter through Decision Engine thresholds
        best_candidate, reason, confidence = self.decision_engine.evaluate_candidates(
            current_intent=resolve_input.intent,
            current_doc_ids=resolve_input.doc_ids,
            candidates_with_scores=candidates_with_scores,
            current_doc_versions=resolve_input.doc_versions
        )

        if best_candidate:
            self.metrics.record_hit()
            return CacheResult(
                hit=True,
                response=best_candidate.response,
                metadata=best_candidate.metadata,
                confidence=confidence
            ), reason
            
        self.metrics.record_miss()
        return CacheResult(hit=False, confidence=confidence), reason

    def store(self, store_input: StoreInput) -> str:
        if store_input.intent == "default":
            store_input.intent = self.intent_classifier.classify(store_input.query)

        if store_input.intent in self.bypass_intents:
            return "skipped_bypass_intent"

        query_vector = self.embedder.embed_query(store_input.query)
        cache_key = self._generate_cache_key(store_input.query)

        entry = CacheEntry(
            query=store_input.query,
            response=store_input.response,
            doc_ids=store_input.doc_ids,
            intent=store_input.intent,
            doc_versions=store_input.doc_versions,
            metadata=store_input.metadata
        )

        # Dual-Write
        self.kv_store.set(cache_key, asdict(entry))
        self.vector_store.upsert(
            ids=[cache_key],
            vectors=[query_vector],
            metadata=[{"query": store_input.query, "intent": store_input.intent}]
        )

        return cache_key
