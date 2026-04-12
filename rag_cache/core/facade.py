from typing import List, Optional, Callable, Dict, Any
import warnings

from rag_cache.core.cache import RetrievalCache, GenerationCache
from rag_cache.core.models import ResolveInput, StoreInput
from rag_cache.core.decision_engine import DecisionEngine
from rag_cache.core.default_intent import RuleBasedIntentClassifier


class UnifiedRAGCache:
    """
    High-level facade for context-aware multi-level RAG caching.

    Provides a simple, plug-and-play API:

        answer = cache.run(query, retriever, llm)

    Handles:
    - L1 Retrieval Cache (query → doc_ids)
    - L2 Generation Cache (query + context → response)
    - Fallback to retriever + LLM
    - Automatic caching
    """

    def __init__(self, use_local_embeddings: bool = True, debug: bool = False):
        self.debug = debug

        # Local imports to keep dependencies optional
        from rag_cache.integrations.key_value_stores.in_memory import InMemoryKeyValueStore
        from rag_cache.integrations.vector_stores.in_memory import InMemoryVectorStore
        # -----------------------------
        # Sync LRU Deletion across DBs
        # -----------------------------
        vector_db = InMemoryVectorStore()

        def sync_eviction(evicted_cache_id: str):
            # Keeps the heavy Vector DB precisely synchronized whenever the KV store fills up!
            if self.debug:
                print(f"[Garbage Collection] LRU evicted Key '{evicted_cache_id}'. Pruning from Vector DB...")
            vector_db.delete(evicted_cache_id)

        kv = InMemoryKeyValueStore(on_evict=sync_eviction)

        # -----------------------------
        # L1: Retrieval Cache
        # -----------------------------
        self.retrieval_layer = RetrievalCache(kv_store=kv)

        # -----------------------------
        # L2: Embedding Setup
        # -----------------------------
        embedder = MockEmbedder()
        if use_local_embeddings:
            try:
                from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder
                embedder = SentenceTransformerEmbedder()
            except ImportError:
                warnings.warn("sentence_transformers not found. Falling back to MockEmbedder.")

        # -----------------------------
        # L2: Generation Cache
        # -----------------------------
        self.generation_layer = GenerationCache(
            embedder=embedder,
            vector_store=vector_db,
            kv_store=kv,
            intent_classifier=RuleBasedIntentClassifier(),
            decision_engine=DecisionEngine()
        )

    # ---------------------------------------------------------
    # 🔥 MAIN ENTRYPOINT (THIS IS WHAT USERS USE)
    # ---------------------------------------------------------
    def run(
        self,
        query: str,
        retriever: Callable[[str], List[str]],
        llm: Callable[[str, List[str]], str]
    ) -> Dict[str, Any]:
        """
        Executes full RAG pipeline with caching.

        Args:
            query: user query
            retriever: function(query) → List[doc_ids]
            llm: function(query, doc_ids) → response

        Returns:
            dict with:
                - answer
                - cache_hit (bool)
                - source ("L1", "L2", "LLM")
        """

        if self.debug:
            print(f"\n[Query]: {query}")

        # -----------------------------
        # Step 1: L1 Retrieval Cache
        # -----------------------------
        doc_ids = self.retrieval_layer.resolve(query)

        if doc_ids:
            if self.debug:
                print("⚡ L1 HIT: Retrieved cached documents")
        else:
            if self.debug:
                print("❌ L1 MISS: Calling retriever")

            doc_ids = retriever(query)
            self.retrieval_layer.store(query, doc_ids)

        # -----------------------------
        # Step 2: L2 Generation Cache
        # -----------------------------
        result, reason = self.generation_layer.resolve(
            ResolveInput(query=query, doc_ids=doc_ids)
        )

        if result.hit:
            if self.debug:
                print(f"⚡ L2 HIT: {reason}")

            return {
                "answer": result.response,
                "cache_hit": True,
                "source": "L2"
            }

        if self.debug:
            print(f"❌ L2 MISS: {reason}")
            print("... Falling back to LLM ...")

        # -----------------------------
        # Step 3: LLM Fallback
        # -----------------------------
        response = llm(query, doc_ids)

        # -----------------------------
        # Step 4: Store in Cache
        # -----------------------------
        self.generation_layer.store(
            StoreInput(query=query, doc_ids=doc_ids, response=response)
        )

        return {
            "answer": response,
            "cache_hit": False,
            "source": "LLM"
        }

    # ---------------------------------------------------------
    # Optional: Metrics passthrough
    # ---------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Returns cache performance metrics."""
        return self.generation_layer.get_stats()