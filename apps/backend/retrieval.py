"""
CryptoAuditAI Retrieval Engine with FastAPI Routers
Consolidated RAG system for transaction analysis
"""

import asyncio
from pathlib import Path
import pickle
import logging
import numpy as np
import faiss
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer

# FastAPI components
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Database - use your existing setup
from database import AsyncSessionLocal, get_db
from models import Transaction

logger = logging.getLogger(__name__)
_lock = asyncio.Lock()

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pydantic Models for API
class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 5

class QueryResult(BaseModel):
    score: float
    transaction_hash: str
    risk_level: str
    value: str
    source: str
    timestamp: datetime

class QueryResponse(BaseModel):
    results: List[QueryResult]
    query: str
    total_count: int

class IndexStatusResponse(BaseModel):
    user_id: int
    has_index: bool
    transaction_count: int
    index_dimensions: Optional[int] = None
    last_updated: datetime

class IndexResponse(BaseModel):
    status: str
    message: str
    user_id: int
    indexed_count: Optional[int] = None

# Create router
router = APIRouter(prefix="/retrieval", tags=["retrieval"])

class RetrievalEngine:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.embedding_model = embedding_model
        self.metadata = {}  # Store metadata per user

    async def _load_index(self, user_id: int):
        """Load (or create) the FAISS index and ID map for a user."""
        try:
            user_dir = Path("indexes") / f"user_{user_id}"
            user_dir.mkdir(parents=True, exist_ok=True)

            index_path = user_dir / "faiss_index.idx"
            id_map_path = user_dir / "id_map.pkl"
            metadata_path = user_dir / "metadata.pkl"

            # Load ID map
            if id_map_path.exists():
                with open(id_map_path, "rb") as f:
                    id_map = pickle.load(f)
            else:
                id_map = {}

            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    self.metadata[user_id] = pickle.load(f)
            else:
                self.metadata[user_id] = []

            # Load or create index
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                if index.d != self.dim:
                    logger.warning(f"Index dimension mismatch for user {user_id}. Creating new index.")
                    index = faiss.IndexFlatIP(self.dim)
            else:
                index = faiss.IndexFlatIP(self.dim)

            return index, id_map
            
        except Exception as e:
            logger.error(f"Failed to load index for user {user_id}: {e}")
            return faiss.IndexFlatIP(self.dim), {}

    async def _save_index(self, user_id: int, index, id_map: dict):
        """Save FAISS index and ID map to disk for persistence."""
        try:
            user_dir = Path("indexes") / f"user_{user_id}"
            user_dir.mkdir(parents=True, exist_ok=True)

            index_path = user_dir / "faiss_index.idx"
            id_map_path = user_dir / "id_map.pkl"
            metadata_path = user_dir / "metadata.pkl"

            # Save index and maps
            faiss.write_index(index, str(index_path))
            with open(id_map_path, "wb") as f:
                pickle.dump(id_map, f)
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata.get(user_id, []), f)
                
            logger.info(f"Index successfully saved for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save index for user {user_id}: {e}")
            raise

    def _tx_row_to_text(self, row):
        """Convert Transaction row to text for embeddings."""
        return f"{row.source} {row.hash} {row.value} risk={row.risk_level} anomaly={row.anomaly_detected}"

    async def index_transactions(self, user_id: int) -> int:
        """Embed and index all new transactions for the user."""
        async with _lock:
            try:
                index, id_map = await self._load_index(user_id)

                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(
                            Transaction.id,
                            Transaction.timestamp,
                            Transaction.source,
                            Transaction.hash,
                            Transaction.value,
                            Transaction.risk_level,
                            Transaction.anomaly_detected,
                        ).where(Transaction.user_id == user_id)
                    )
                    rows = result.all()

                if not rows:
                    logger.info(f"No transactions found for user {user_id}")
                    return 0

                existing_ids = set(id_map.values())
                new_texts, new_ids = [], []

                for row in rows:
                    tx_id = str(row.id)
                    if tx_id in existing_ids:
                        continue
                    new_texts.append(self._tx_row_to_text(row))
                    new_ids.append(tx_id)

                if not new_texts:
                    logger.info(f"No new transactions to index for user {user_id}")
                    return 0

                # Embed and add to FAISS
                embeddings = self.embedding_model.encode(new_texts)
                vecs = np.array(embeddings).astype("float32")
                index.add(vecs)

                # Update ID map and metadata
                start_idx = len(id_map)
                for i, tx_id in enumerate(new_ids):
                    id_map[start_idx + i] = tx_id
                
                # Store metadata for each transaction
                for row in rows:
                    if str(row.id) in new_ids:
                        self.metadata[user_id].append({
                            "tx_id": row.id,
                            "risk_score": row.risk_level,
                            "hash": row.hash,
                            "timestamp": row.timestamp
                        })

                await self._save_index(user_id, index, id_map)
                logger.info(f"Successfully indexed {len(new_ids)} transactions for user {user_id}")
                return len(new_ids)
                
            except Exception as e:
                logger.error(f"Indexing failed for user {user_id}: {e}")
                return 0

    async def query(self, user_id: int, query_text: str, top_k: int = 5):
        """Retrieve top-k relevant transactions for a user based on query."""
        try:
            index, id_map = await self._load_index(user_id)
            user_metadata = self.metadata.get(user_id, [])
            
            if len(id_map) == 0 or index.ntotal == 0:
                logger.warning(f"No indexed data found for user {user_id}")
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query_text])
            query_embedding_array = np.array(query_embedding).astype("float32")

            # Perform search
            distances, indices = index.search(query_embedding_array, top_k)

            results = []
            async with AsyncSessionLocal() as session:
                for i, idx in enumerate(indices[0]):
                    if idx == -1 or idx >= len(user_metadata):
                        continue
                    
                    # Get transaction ID from metadata
                    tx_id = user_metadata[idx]["tx_id"]
                    transaction = await session.get(Transaction, tx_id)
                    
                    if transaction:
                        # Convert distance to similarity score (higher is better)
                        similarity = 1.0 / (1.0 + distances[0][i])
                        results.append((similarity, transaction))

            return results
            
        except Exception as e:
            logger.error(f"Query failed for user {user_id}: {e}")
            return []

# API Endpoints
@router.post("/users/{user_id}/index", response_model=IndexResponse)
async def index_user_transactions(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Trigger background indexing of user transactions"""
    engine = RetrievalEngine()
    
    async def index_task():
        try:
            count = await engine.index_transactions(user_id)
            logger.info(f"Background indexing completed for user {user_id}: {count} transactions indexed")
        except Exception as e:
            logger.error(f"Background indexing failed for user {user_id}: {e}")
    
    background_tasks.add_task(index_task)
    return IndexResponse(
        status="accepted",
        message="Indexing started in background",
        user_id=user_id
    )

@router.post("/users/{user_id}/query", response_model=QueryResponse)
async def query_transactions(
    user_id: int,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Query user's transactions using semantic search"""
    try:
        engine = RetrievalEngine()
        results = await engine.query(user_id, query_request.query_text, query_request.top_k)
        
        formatted_results = []
        for score, transaction in results:
            formatted_results.append(QueryResult(
                score=float(score),
                transaction_hash=transaction.hash,
                risk_level=transaction.risk_level,
                value=transaction.value,
                source=transaction.source,
                timestamp=transaction.timestamp
            ))
        
        return QueryResponse(
            results=formatted_results,
            query=query_request.query_text,
            total_count=len(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Query failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/users/{user_id}/index/status", response_model=IndexStatusResponse)
async def get_index_status(user_id: int):
    """Check if user has an index and get basic stats"""
    try:
        engine = RetrievalEngine()
        index, id_map = await engine._load_index(user_id)
        
        return IndexStatusResponse(
            user_id=user_id,
            has_index=index.ntotal > 0,
            transaction_count=index.ntotal,
            index_dimensions=index.d if hasattr(index, 'd') else None,
            last_updated=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/users/{user_id}/index")
async def delete_user_index(user_id: int):
    """Delete a user's index (for cleanup)"""
    try:
        import shutil
        user_dir = Path("indexes") / f"user_{user_id}"
        
        if user_dir.exists():
            shutil.rmtree(user_dir)
            logger.info(f"Deleted index for user {user_id}")
            return {"status": "deleted", "user_id": user_id}
        else:
            return {"status": "not_found", "user_id": user_id}
            
    except Exception as e:
        logger.error(f"Failed to delete index for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {str(e)}")

# Utility function for scheduled indexing
async def index_all_users_transactions():
    """Index transactions for all users (for scheduler)"""
    engine = RetrievalEngine()
    
    async with AsyncSessionLocal() as session:
        # Get all distinct user IDs with transactions
        result = await session.execute(
            select(Transaction.user_id).distinct()
        )
        user_ids = result.scalars().all()
    
    for user_id in user_ids:
        try:
            count = await engine.index_transactions(user_id)
            if count > 0:
                logger.info(f"Scheduled indexing: {count} new transactions for user {user_id}")
        except Exception as e:
            logger.error(f"Scheduled indexing failed for user {user_id}: {e}")

# Export the router for main.py
__all__ = ['router', 'RetrievalEngine', 'index_all_users_transactions']