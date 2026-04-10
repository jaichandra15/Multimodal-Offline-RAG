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
        # Ollama is always used for embeddings (vector search).
        self.ollama = ollama_client

        # ── LLM backend selection ──────────────────────────────────────────
        if settings.llm_backend == "gemini":
            from backend.core.gemini_client import get_gemini_client
            self._llm = get_gemini_client()
            self._llm_label = f"gemini/{settings.gemini_model}"
            logger.info(
                f"RAGEngine: using Gemini backend ({settings.gemini_model}) "
                "for LLM inference; Ollama used for embeddings only."
            )
        else:
            self._llm = ollama_client  # type: ignore[assignment]
            self._llm_label = settings.ollama_llm_model
            logger.info(
                f"RAGEngine: using Ollama backend ({settings.ollama_llm_model}) "
                "for LLM inference."
            )

        self.max_context_length = 6000   # ↑ from 3000 — gives model more context
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
    
    def _format_context(self, search_results) -> str:
        """
        Format retrieved chunks into a clean, clearly-delimited context block.
        Each source gets a numbered header so the model can cite them easily.
        """
        if not search_results:
            return "No relevant information was found in the knowledge base."

        parts = []
        for i, result in enumerate(search_results, 1):
            parts.append(
                f"--- Source {i}: {result.document_title} ---\n"
                f"{result.content.strip()}"
            )
        raw = "\n\n".join(parts)

        # Truncate if necessary
        if len(raw) > self.max_context_length:
            raw = raw[: self.max_context_length] + "\n\n...[context truncated]..."
        return raw

    def _build_prompt(
        self,
        user_query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a well-structured prompt for Ollama with:
        - A detailed system persona with formatting rules
        - Optional conversation history for memory
        - The retrieved context, clearly delimited
        - The user question
        """
        # ── System instructions ────────────────────────────────────────────────
        system = """You are an expert AI assistant with access to a knowledge base.

FORMATTING RULES (follow these strictly):
- Use **Markdown** formatting in every response.
- Structure your answer with a short **introductory sentence**, then use:
  - `##` headers to separate distinct topics if the answer covers multiple areas
  - Bullet points (`-`) or numbered lists for enumerations, steps, or comparisons
  - `**bold**` for key terms and important facts
  - Code blocks (``` ```) for any code, commands, or technical syntax
- Keep each bullet point concise — one idea per bullet.
- After the main answer, add a **Summary** section (1–2 sentences) only if the response is long.
- Do NOT write long unbroken paragraphs. Break text up.
- If the knowledge base does not contain enough information to answer fully, say so clearly.
- Do NOT fabricate information not present in the provided sources."""

        # ── Conversation history (for multi-turn memory) ───────────────────────
        history_block = ""
        if conversation_history:
            history_lines = []
            # Only include last 6 turns to avoid prompt bloat
            for msg in conversation_history[-6:]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_lines.append(f"{role_label}: {msg['content']}")
            history_block = (
                "\n\n--- Conversation History ---\n"
                + "\n".join(history_lines)
                + "\n--- End of History ---"
            )

        # ── Final prompt assembly ──────────────────────────────────────────────
        prompt = f"""{system}{history_block}

--- Knowledge Base Context ---
{context}
--- End of Context ---

User Question: {user_query}

Answer (use Markdown formatting as instructed above):"""

        return prompt

    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search knowledge base for relevant chunks.
        
        Uses hybrid search (vector + keyword) by default for better results.
        
        Args:
            session: Database session
            query: Search query
            limit: Maximum number of results
            use_hybrid: Override hybrid search setting (default: self.use_hybrid_search)
            metadata_filter: Optional JSONB containment filter applied to chunk
                metadata. Only chunks whose metadata contains ALL specified
                key-value pairs are returned. E.g. {"page": 3} or
                {"file_path": "report.pdf"}.
            
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
                limit=fetch_limit,
                metadata_filter=metadata_filter
            )
        else:
            logger.info("Using vector-only search...")
            results = await vector_search(
                session,
                query_embedding,
                limit=fetch_limit,
                metadata_filter=metadata_filter
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
    
    async def search_detailed(
        self,
        session: AsyncSession,
        query: str,
        limit: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search with full pipeline trace for real-time visualization.
        
        Returns dict with keys:
          results     — final SearchResult list (same as search())
          trace       — dict describing the retrieval pipeline
        """
        t0 = time.time()
        
        # 1. Embed
        t_embed = time.time()
        query_embedding = await self.ollama.generate_embedding(query)
        embed_ms = round((time.time() - t_embed) * 1000, 1)
        
        # 2. Retrieve
        should_hybrid = self.use_hybrid_search
        search_method = "hybrid" if should_hybrid else "vector"
        fetch_limit = limit or settings.top_k_results
        if self.use_reranker and settings.reranker_top_k > fetch_limit:
            fetch_limit = settings.reranker_top_k
        
        t_retrieval = time.time()
        if should_hybrid:
            raw_results = await hybrid_search(
                session, query=query, query_embedding=query_embedding,
                limit=fetch_limit, metadata_filter=metadata_filter
            )
        else:
            raw_results = await vector_search(
                session, query_embedding, limit=fetch_limit,
                metadata_filter=metadata_filter
            )
        retrieval_ms = round((time.time() - t_retrieval) * 1000, 1)
        
        # Snapshot pre-rerank
        pre_rerank = [
            {
                "rank": i + 1,
                "chunk_id": str(r.chunk_id),
                "doc_title": r.document_title,
                "preview": r.content[:180],
                "score": round(r.similarity, 4),
            }
            for i, r in enumerate(raw_results)
        ]
        
        # 3. Rerank
        reranked = False
        rerank_model = None
        rerank_ms = None
        final = raw_results
        
        if self.use_reranker and raw_results:
            try:
                reranker = self._get_reranker()
                if reranker:
                    rerank_model = settings.reranker_model
                    t_rr = time.time()
                    final_limit = limit or settings.top_k_results
                    final = reranker.rerank(query, raw_results, top_k=final_limit)
                    rerank_ms = round((time.time() - t_rr) * 1000, 1)
                    reranked = True
                    metrics.reranker_calls_total.labels(status="success").inc()
            except Exception as e:
                logger.error(f"Reranking failed in detailed search: {e}")
                metrics.reranker_calls_total.labels(status="error").inc()
        
        if not reranked and limit and len(final) > limit:
            final = final[:limit]
        
        # Snapshot post-rerank
        post_rerank = [
            {
                "rank": i + 1,
                "chunk_id": str(r.chunk_id),
                "doc_title": r.document_title,
                "preview": r.content[:180],
                "score": round(r.similarity, 4),
            }
            for i, r in enumerate(final)
        ]
        
        total_ms = round((time.time() - t0) * 1000, 1)
        
        # Record metrics
        metrics.rag_search_latency.labels(method=search_method).observe(total_ms / 1000)
        metrics.rag_chunks_retrieved.observe(len(final))
        
        return {
            "results": final,
            "trace": {
                "search_method": search_method,
                "embed_ms": embed_ms,
                "retrieval_ms": retrieval_ms,
                "reranked": reranked,
                "rerank_model": rerank_model,
                "rerank_ms": rerank_ms,
                "candidates_fetched": len(pre_rerank),
                "final_count": len(post_rerank),
                "total_ms": total_ms,
                "pre_rerank": pre_rerank,
                "post_rerank": post_rerank,
                "metadata_filter": metadata_filter,
            },
        }
    
    async def generate_answer(
        self,
        session: AsyncSession,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        search_results: Optional[List[SearchResult]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate answer to user query using RAG.

        Args:
            session: Database session
            query: User's question
            conversation_history: Optional conversation history
            search_results: Pre-fetched search results. If provided, the search
                step is skipped entirely — use this to avoid double-searching
                when the caller already has results (e.g. chat()).
            metadata_filter: Optional JSONB metadata filter forwarded to the
                search step (ignored when search_results is provided).

        Returns:
            Generated answer
        """
        # Only search if results were not provided by the caller.
        if search_results is None:
            search_results = await self.search(
                session, query, metadata_filter=metadata_filter
            )

        # Format context using the shared helper
        context = self._format_context(search_results)

        # Build prompt
        prompt = self._build_prompt(query, context, conversation_history)

        # Generate response
        logger.info(f"Generating response via {self._llm_label}...")
        start_time = time.time()
        answer = await self._llm.generate_chat_completion(prompt)
        generation_duration = time.time() - start_time

        # Record metrics
        metrics.rag_generation_latency.labels(model=self._llm_label).observe(generation_duration)
        logger.info(f"Generated response in {generation_duration:.2f}s")

        return answer
    
    async def generate_answer_stream(
        self,
        session: AsyncSession,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        search_results: Optional[List[SearchResult]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer with streaming.

        Args:
            session: Database session
            query: User's question
            conversation_history: Optional conversation history
            search_results: Pre-fetched search results. If provided, the search
                step is skipped entirely — avoids double-searching in the
                streaming route which already calls search_detailed().
            metadata_filter: Optional JSONB metadata filter forwarded to the
                search step (ignored when search_results is provided).

        Yields:
            Text chunks as they are generated
        """
        # Only search if results were not provided by the caller.
        if search_results is None:
            search_results = await self.search(
                session, query, metadata_filter=metadata_filter
            )

        # Format context using the shared helper
        context = self._format_context(search_results)

        # Build prompt
        prompt = self._build_prompt(query, context, conversation_history)

        # Stream response with timing
        logger.info(f"Streaming response via {self._llm_label}...")
        start_time = time.time()
        async for chunk in self._llm.generate_chat_completion_stream(prompt):
            yield chunk

        # Record generation metrics
        generation_duration = time.time() - start_time
        metrics.rag_generation_latency.labels(model=self._llm_label).observe(generation_duration)
        logger.info(f"Generation completed in {generation_duration:.2f}s")
    
    async def chat(
        self,
        session: AsyncSession,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Complete chat interaction with RAG.

        Args:
            session: Database session
            user_query: User's message
            conversation_history: Optional conversation history
            metadata_filter: Optional JSONB metadata filter applied to the
                retrieval step. Only chunks whose metadata contains ALL
                specified key-value pairs will be included in context.

        Returns:
            Dictionary with response, citations, and updated conversation history
        """
        # Search once — pass results to generate_answer to avoid a second search.
        search_results = await self.search(
            session, user_query, metadata_filter=metadata_filter
        )

        # Generate answer, reusing the results fetched above.
        answer = await self.generate_answer(
            session,
            user_query,
            conversation_history,
            search_results=search_results,
        )

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
