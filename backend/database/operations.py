"""
Database operations - CRUD and vector search functions.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, func, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database.models import Document, Chunk
from backend.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Document Operations
# ============================================================================

async def create_document(
    session: AsyncSession,
    title: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Document:
    """
    Create a new document.
    
    Args:
        session: Database session
        title: Document title
        source: Document source path
        content: Full document content
        metadata: Optional metadata dictionary
        
    Returns:
        Created Document instance
    """
    document = Document(
        title=title,
        source=source,
        content=content,
        metadata_=metadata or {}
    )
    session.add(document)
    await session.flush()
    return document


async def get_document(session: AsyncSession, document_id: UUID) -> Optional[Document]:
    """
    Get document by ID.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        Document instance or None
    """
    result = await session.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def list_documents(
    session: AsyncSession,
    limit: int = 100,
    offset: int = 0
) -> List[Document]:
    """
    List all documents with pagination.
    
    Args:
        session: Database session
        limit: Maximum number of results
        offset: Number of results to skip
        
    Returns:
        List of Document instances
    """
    result = await session.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, document_id: UUID) -> bool:
    """
    Delete a document and all its chunks.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        True if deleted, False if not found
    """
    result = await session.execute(
        delete(Document).where(Document.id == document_id)
    )
    return result.rowcount > 0


async def delete_all_documents(session: AsyncSession) -> int:
    """
    Delete all documents and chunks.
    
    Args:
        session: Database session
        
    Returns:
        Number of documents deleted
    """
    result = await session.execute(delete(Document))
    return result.rowcount


async def get_document_count(session: AsyncSession) -> int:
    """
    Get total number of documents.
    
    Args:
        session: Database session
        
    Returns:
        Document count
    """
    result = await session.execute(select(func.count(Document.id)))
    return result.scalar_one()


# ============================================================================
# Chunk Operations
# ============================================================================

async def create_chunk(
    session: AsyncSession,
    document_id: UUID,
    content: str,
    embedding: List[float],
    chunk_index: int,
    token_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk:
    """
    Create a new chunk.
    
    Args:
        session: Database session
        document_id: Parent document UUID
        content: Chunk text content
        embedding: Embedding vector
        chunk_index: Index of chunk in document
        token_count: Optional token count
        metadata: Optional metadata dictionary
        
    Returns:
        Created Chunk instance
    """
    chunk = Chunk(
        document_id=document_id,
        content=content,
        embedding=embedding,
        chunk_index=chunk_index,
        token_count=token_count,
        metadata_=metadata or {}
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def get_chunk_count(session: AsyncSession) -> int:
    """
    Get total number of chunks.
    
    Args:
        session: Database session
        
    Returns:
        Chunk count
    """
    result = await session.execute(select(func.count(Chunk.id)))
    return result.scalar_one()


async def get_chunks_by_document(
    session: AsyncSession,
    document_id: UUID
) -> List[Chunk]:
    """
    Get all chunks for a document.
    
    Args:
        session: Database session
        document_id: Document UUID
        
    Returns:
        List of Chunk instances
    """
    result = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    return list(result.scalars().all())


# ============================================================================
# Vector Search Operations
# ============================================================================

class SearchResult:
    """Container for search results with metadata."""
    
    def __init__(
        self,
        chunk_id: UUID,
        document_id: UUID,
        content: str,
        similarity: float,
        chunk_metadata: Dict[str, Any],
        document_title: str,
        document_source: str
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.similarity = similarity
        self.chunk_metadata = chunk_metadata
        self.document_title = document_title
        self.document_source = document_source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.chunk_metadata,
            "document_title": self.document_title,
            "document_source": self.document_source
        }


async def vector_search(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = None,
    similarity_threshold: float = None
) -> List[SearchResult]:
    """
    Perform vector similarity search using cosine distance.
    
    Args:
        session: Database session
        query_embedding: Query embedding vector
        limit: Maximum number of results (defaults to settings.top_k_results)
        similarity_threshold: Minimum similarity score (defaults to settings.similarity_threshold)
        
    Returns:
        List of SearchResult instances ordered by similarity (descending)
    """
    if limit is None:
        limit = settings.top_k_results
    
    if similarity_threshold is None:
        similarity_threshold = settings.similarity_threshold
    
    # Use pgvector's cosine distance operator (<=>)
    # Similarity = 1 - distance (so higher is better)
    query = text("""
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> :query_embedding) AS similarity,
            c.metadata,
            d.title AS document_title,
            d.source AS document_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
            AND 1 - (c.embedding <=> :query_embedding) >= :threshold
        ORDER BY c.embedding <=> :query_embedding
        LIMIT :limit
    """)
    
    # Convert embedding to string format for pgvector
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    logger.info(f"Vector search: threshold={similarity_threshold}, limit={limit}")
    
    result = await session.execute(
        query,
        {
            "query_embedding": embedding_str,
            "threshold": similarity_threshold,
            "limit": limit
        }
    )
    
    rows = result.fetchall()
    
    logger.info(f"Vector search returned {len(rows)} results")
    for row in rows:
        logger.info(f"  - {row.document_title}: similarity={row.similarity:.4f}")
    
    return [
        SearchResult(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            content=row.content,
            similarity=float(row.similarity),
            chunk_metadata=row.metadata,
            document_title=row.document_title,
            document_source=row.document_source
        )
        for row in rows
    ]


def _extract_search_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from a query for fuzzy search.
    
    Removes common stop words and short words that don't add value.
    
    Args:
        query: User's search query
        
    Returns:
        List of keywords to search for
    """
    stop_words = {
        'what', 'where', 'when', 'which', 'that', 'this', 'there', 'their',
        'have', 'been', 'were', 'from', 'with', 'they', 'them', 'then',
        'than', 'these', 'those', 'will', 'would', 'could', 'should',
        'about', 'after', 'before', 'between', 'into', 'through',
        'during', 'above', 'below', 'does', 'doing', 'each', 'some',
        'most', 'other', 'such', 'only', 'same', 'also', 'back',
        'being', 'here', 'just', 'over', 'under', 'again', 'further',
        'once', 'more', 'very', 'your', 'yours', 'yourself', 'what\'s',
        'how', 'why', 'who', 'whom', 'the', 'and', 'but', 'for', 'are', 'was'
    }
    
    words = []
    for word in query.split():
        # Clean the word
        clean_word = word.strip('?.,!;:"\'()[]{}').lower()
        # Keep words that are meaningful
        if len(clean_word) >= 3 and clean_word not in stop_words:
            words.append(clean_word)
    
    return words


async def keyword_search(
    session: AsyncSession,
    query: str,
    limit: int = None,
    similarity_threshold: float = 0.3
) -> List[SearchResult]:
    """
    Perform fuzzy keyword search using PostgreSQL's pg_trgm extension.
    
    Uses word_similarity for fuzzy matching that handles:
    - OCR errors and typos (e.g., "Implementation" vs "Implementaon")
    - Partial word matches within text
    - Character transpositions
    
    Falls back to ILIKE if trigram search fails.
    
    Args:
        session: Database session
        query: Search query text
        limit: Maximum number of results
        similarity_threshold: Minimum trigram similarity (0.0-1.0, default 0.3)
        
    Returns:
        List of SearchResult instances ordered by relevance
    """
    if limit is None:
        limit = settings.top_k_results
    
    # Extract meaningful keywords from query
    keywords = _extract_search_keywords(query)
    
    if not keywords:
        logger.info("No meaningful keywords found in query")
        return []
    
    logger.info(f"Keyword search for: {query[:50]}... (keywords: {keywords[:5]})")
    
    # Use word_similarity which finds the best matching word within the text
    # This is better than similarity() for searching within longer text
    try:
        # Set the word similarity threshold
        await session.execute(text(f"SET pg_trgm.word_similarity_threshold = {similarity_threshold}"))
        
        # Build conditions for each keyword using ILIKE for partial matching
        # combined with word_similarity scoring for ranking
        keyword_conditions = []
        similarity_scores = []
        params = {"limit": limit, "num_keywords": float(len(keywords))}
        
        for i, kw in enumerate(keywords):
            # Use ILIKE with prefix to handle OCR issues (match first 4+ chars)
            prefix = kw[:4] if len(kw) >= 4 else kw
            keyword_conditions.append(f"c.content ILIKE :pattern{i}")
            similarity_scores.append(f"word_similarity(:kw{i}, c.content)")
            params[f"kw{i}"] = kw
            params[f"pattern{i}"] = f"%{prefix}%"
        
        where_clause = " OR ".join(keyword_conditions)
        score_clause = " + ".join(similarity_scores)
        
        fuzzy_query = text(f"""
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                c.content,
                ({score_clause}) / :num_keywords AS rank,
                c.metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :limit
        """)
        
        result = await session.execute(fuzzy_query, params)
        rows = result.fetchall()
        
        logger.info(f"Fuzzy keyword search returned {len(rows)} results")
            
    except Exception as e:
        # Fallback to simple ILIKE search if fuzzy fails
        logger.warning(f"Fuzzy search failed: {e}, falling back to ILIKE")
        
        # Build ILIKE conditions for each keyword
        ilike_conditions = []
        params = {"limit": limit}
        for i, kw in enumerate(keywords[:3]):  # Limit to first 3 keywords
            ilike_conditions.append(f"c.content ILIKE :pattern{i}")
            params[f"pattern{i}"] = f"%{kw}%"
        
        where_clause = " OR ".join(ilike_conditions) if ilike_conditions else "TRUE"
        
        fallback_query = text(f"""
            SELECT 
                c.id AS chunk_id,
                c.document_id,
                c.content,
                1.0 AS rank,
                c.metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
            LIMIT :limit
        """)
        
        result = await session.execute(fallback_query, params)
        rows = result.fetchall()
        logger.info(f"Fallback ILIKE search returned {len(rows)} results")
    
    return [
        SearchResult(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            content=row.content,
            similarity=float(row.rank) if row.rank else 0.0,
            chunk_metadata=row.metadata,
            document_title=row.document_title,
            document_source=row.document_source
        )
        for row in rows
    ]


def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    keyword_results: List[SearchResult],
    k: int = 60,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> List[SearchResult]:
    """
    Combine vector and keyword search results using Reciprocal Rank Fusion (RRF).
    
    RRF is a simple but effective method to combine multiple ranked lists.
    Score = sum(weight / (k + rank)) for each list where the item appears.
    
    Args:
        vector_results: Results from vector search
        keyword_results: Results from keyword search
        k: RRF constant (default 60, higher = more weight to lower ranks)
        vector_weight: Weight for vector search results
        keyword_weight: Weight for keyword search results
        
    Returns:
        Combined and re-ranked list of SearchResult
    """
    # Create a map of chunk_id -> (result, rrf_score)
    scores: Dict[UUID, tuple] = {}
    
    # Add vector search scores
    for rank, result in enumerate(vector_results):
        rrf_score = vector_weight / (k + rank + 1)
        scores[result.chunk_id] = (result, rrf_score)
    
    # Add keyword search scores
    for rank, result in enumerate(keyword_results):
        rrf_score = keyword_weight / (k + rank + 1)
        if result.chunk_id in scores:
            # Combine scores if result appears in both lists
            existing_result, existing_score = scores[result.chunk_id]
            scores[result.chunk_id] = (existing_result, existing_score + rrf_score)
        else:
            scores[result.chunk_id] = (result, rrf_score)
    
    # Sort by combined RRF score and return
    sorted_results = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    
    # Update similarity scores to reflect RRF ranking
    final_results = []
    for result, rrf_score in sorted_results:
        result.similarity = rrf_score  # Replace with RRF score for transparency
        final_results.append(result)
    
    return final_results


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_embedding: List[float],
    limit: int = None,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> List[SearchResult]:
    """
    Perform hybrid search combining vector similarity and keyword matching.
    
    This approach combines:
    1. Vector search (semantic similarity via embeddings)
    2. Keyword search (lexical matching via PostgreSQL full-text search)
    
    Results are combined using Reciprocal Rank Fusion (RRF).
    
    Args:
        session: Database session
        query: Original query text (for keyword search)
        query_embedding: Query embedding vector (for vector search)
        limit: Maximum number of final results
        vector_weight: Weight for vector results (default 0.6)
        keyword_weight: Weight for keyword results (default 0.4)
        
    Returns:
        Combined list of SearchResult instances
    """
    if limit is None:
        limit = settings.top_k_results
    
    # Fetch more results initially for better fusion
    fetch_limit = limit * 3
    
    logger.info(f"Hybrid search: vector_weight={vector_weight}, keyword_weight={keyword_weight}")
    
    # Perform both searches
    vector_results = await vector_search(session, query_embedding, limit=fetch_limit)
    keyword_results = await keyword_search(session, query, limit=fetch_limit)
    
    logger.info(f"Vector search: {len(vector_results)} results, Keyword search: {len(keyword_results)} results")
    
    # Combine using RRF
    combined_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight
    )
    
    # Return top results
    final_results = combined_results[:limit]
    logger.info(f"Hybrid search returning {len(final_results)} results")
    
    return final_results


async def search_knowledge_base(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = None
) -> str:
    """
    Search knowledge base and format results as context string.
    
    Args:
        session: Database session
        query_embedding: Query embedding vector
        limit: Maximum number of results
        
    Returns:
        Formatted context string with sources
    """
    results = await vector_search(session, query_embedding, limit=limit)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results with sources
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {result.document_title}]\n{result.content}"
        )
    
    return "\n\n".join(context_parts)
