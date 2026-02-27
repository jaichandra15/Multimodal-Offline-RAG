"""
RAG Engine - Core logic for retrieval-augmented generation.
Combines vector search with Ollama LLM to answer questions.
"""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.ollama_client import ollama_client
from backend.database.operations import vector_search, hybrid_search, SearchResult
from backend.config import settings
from backend.core.observability import metrics

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG Engine for knowledge-based question answering."""
    
    def __init__(self):
        """Initialize RAG engine."""
        self.ollama = ollama_client
        self.max_context_length = 3000
        self.use_hybrid_search = settings.use_hybrid_search
        self.use_reranker = settings.reranker_enabled
        self._reranker = None  # Lazy load reranker
    
    def _get_reranker(self):
        """Get or create reranker instance (lazy loading)."""
        if self._reranker is None and self.use_reranker:
            try:
                from backend.core.reranker import get_reranker
                self._reranker = get_reranker()
                logger.info("Reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.use_reranker = False
        return self._reranker
    
    def _build_prompt(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build prompt for Ollama with context and query.
        
        Args:
            user_query: User's question
            context: Retrieved context from knowledge base
            conversation_history: Optional conversation history
            
        Returns:
            Formatted prompt
        """
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "\n...(context truncated)"
        
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
Synthesize the information into a clear, natural answer. DO NOT just list the sources - provide a coherent response.

Context from knowledge base:
{context}

User Question: {user_query}

Your Answer (synthesize the information above into a clear response):"""
        
        return prompt
    
    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: Optional[int] = None,
        use_hybrid: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        Search knowledge base for relevant chunks.
        
        Uses hybrid search (vector + keyword) by default for better results.
        
        Args:
            session: Database session
            query: Search query
            limit: Maximum number of results
            use_hybrid: Override hybrid search setting (default: self.use_hybrid_search)
            
        Returns:
            List of SearchResult instances
        """
        start_time = time.time()
        
        # Generate embedding for query
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = await self.ollama.generate_embedding(query)
        
        # Determine search method
        should_use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid_search
        search_method = "hybrid" if should_use_hybrid else "vector"
        
        # Determine how many candidates to fetch (more if reranking)
        fetch_limit = limit or settings.top_k_results
        if self.use_reranker and settings.reranker_top_k > fetch_limit:
            fetch_limit = settings.reranker_top_k
        
        # Search vector database
        if should_use_hybrid:
            logger.info("Using hybrid search (vector + keyword)...")
            results = await hybrid_search(
                session,
                query=query,
                query_embedding=query_embedding,
                limit=fetch_limit
            )
        else:
            logger.info("Using vector-only search...")
            results = await vector_search(
                session,
                query_embedding,
                limit=fetch_limit
            )
        
        # Apply reranking if enabled
        if self.use_reranker and results:
            try:
                reranker = self._get_reranker()
                if reranker:
                    logger.info(f"Reranking {len(results)} results...")
                    final_limit = limit or settings.top_k_results
                    results = reranker.rerank(query, results, top_k=final_limit)
                    metrics.reranker_calls_total.labels(status="success").inc()
                else:
                    metrics.reranker_calls_total.labels(status="disabled").inc()
            except Exception as e:
                logger.error(f"Reranking failed: {e}", exc_info=True)
                metrics.reranker_calls_total.labels(status="error").inc()
                # Continue with original results
                if limit and len(results) > limit:
                    results = results[:limit]
        elif limit and len(results) > limit:
            results = results[:limit]
        
        # Record metrics
        search_duration = time.time() - start_time
        metrics.rag_search_latency.labels(method=search_method).observe(search_duration)
        metrics.rag_chunks_retrieved.observe(len(results))
        
        logger.info(f"Found {len(results)} relevant chunks in {search_duration:.2f}s")
        return results
    
    async def generate_answer(
        self,
        session: AsyncSession,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate answer to user query using RAG.
        
        Args:
            session: Database session
            query: User's question
            conversation_history: Optional conversation history
            
        Returns:
            Generated answer
        """
        # Search knowledge base
        search_results = await self.search(session, query)
        
        # Format context
        if not search_results:
            context = "No relevant information found in the knowledge base."
        else:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"[Source {i}: {result.document_title}]\n{result.content}"
                )
            context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_prompt(query, context, conversation_history)
        
        # Generate response
        logger.info("Generating response...")
        start_time = time.time()
        answer = await self.ollama.generate_chat_completion(prompt)
        generation_duration = time.time() - start_time
        
        # Record metrics
        metrics.rag_generation_latency.labels(model=settings.ollama_llm_model).observe(generation_duration)
        logger.info(f"Generated response in {generation_duration:.2f}s")
        
        return answer
    
    async def generate_answer_stream(
        self,
        session: AsyncSession,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer with streaming.
        
        Args:
            session: Database session
            query: User's question
            conversation_history: Optional conversation history
            
        Yields:
            Text chunks as they are generated
        """
        # Search knowledge base
        search_results = await self.search(session, query)
        
        # Format context
        if not search_results:
            context = "No relevant information found in the knowledge base."
        else:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"[Source {i}: {result.document_title}]\n{result.content}"
                )
            context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_prompt(query, context, conversation_history)
        
        # Stream response with timing
        logger.info("Streaming response...")
        start_time = time.time()
        async for chunk in self.ollama.generate_chat_completion_stream(prompt):
            yield chunk
        
        # Record generation metrics
        generation_duration = time.time() - start_time
        metrics.rag_generation_latency.labels(model=settings.ollama_llm_model).observe(generation_duration)
        logger.info(f"Generation completed in {generation_duration:.2f}s")
    
    async def chat(
        self,
        session: AsyncSession,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Complete chat interaction with RAG.
        
        Args:
            session: Database session
            user_query: User's message
            conversation_history: Optional conversation history
            
        Returns:
            Dictionary with response, citations, and updated conversation history
        """
        # Search knowledge base for citations
        search_results = await self.search(session, user_query)
        
        # Generate answer
        answer = await self.generate_answer(session, user_query, conversation_history)
        
        # Record success metric
        metrics.rag_requests_total.labels(status="success").inc()
        
        # Build citations from search results
        citations = []
        for i, result in enumerate(search_results, 1):
            citations.append({
                "number": i,
                "chunk_id": str(result.chunk_id),
                "document_id": str(result.document_id),
                "document_title": result.document_title,
                "document_source": result.document_source,
                "content": result.content,
                "metadata": result.chunk_metadata,
                "similarity": result.similarity
            })
        
        # Update conversation history
        if conversation_history is None:
            conversation_history = []
        
        updated_history = conversation_history.copy()
        updated_history.append({"role": "user", "content": user_query})
        updated_history.append({"role": "assistant", "content": answer})
        
        return {
            "response": answer,
            "conversation_history": updated_history,
            "citations": citations
        }


# Global RAG engine instance
rag_engine = RAGEngine()
