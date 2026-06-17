import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rag_cache import RAGCache

app = FastAPI(title="RAGCache Load Testing Server")

# Initialize RAGCache (Prometheus metrics will be registered on the default registry)
use_local = os.environ.get("USE_LOCAL_EMBEDDINGS", "False").lower() in ("true", "1", "yes")
cache = RAGCache(use_local_embeddings=use_local, debug=False)

class QueryRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    scope: str = "tenant"
    doc_ids: List[str] = ["doc_1", "doc_2"]

def mock_llm(query: str, doc_ids: List[str]) -> str:
    return f"Response for: {query}"

@app.post("/query")
async def run_query(req: QueryRequest):
    try:
        res = cache.run(
            query=req.query,
            retriever=lambda q: req.doc_ids,
            llm=mock_llm,
            tenant_id=req.tenant_id,
            scope=req.scope
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Exposes RAGCache Prometheus metrics for scraper consumption."""
    return Response(
        content=generate_latest(), 
        media_type=CONTENT_TYPE_LATEST,
        headers={"X-PID": str(os.getpid())}
    )

@app.get("/health")
async def health():
    return {"status": "ok"}
