"""
API routes for the RAG backend.
"""

import asyncio
import logging
import json
import os
import tempfile
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    Citation,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    DocumentListResponse,
    DocumentInfo,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    FileUploadResponse,
    # RAGAS
    RAGASEvaluationRequest,
    RAGASEvaluationResponse,
    RAGASScores,
    RAGASHistoryResponse,
)
from backend.database.connection import get_db_session, db_manager
from backend.database.operations import (
    get_document_count,
    get_chunk_count,
    list_documents,
    create_document,
    create_chunk
)
from backend.core.rag_engine import rag_engine
from backend.core.ollama_client import ollama_client
from backend.config import settings
from backend.ingestion.pipeline import IngestionPipeline
from backend.ingestion.chunker import DoclingHybridChunker, ChunkingConfig
from backend.ingestion.embedder import OllamaEmbedder

try:
    from backend.core.observability import metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(session: AsyncSession = Depends(get_db_session)):
    """
    Check system health status.
    
    Returns status of database, Ollama, and knowledge base statistics.
    """
    try:
        db_healthy = await db_manager.health_check()
        ollama_healthy = await ollama_client.health_check()
        
        # Get knowledge base stats
        doc_count = await get_document_count(session)
        chunk_count = await get_chunk_count(session)
        
        return HealthResponse(
            status="healthy" if (db_healthy and ollama_healthy) else "degraded",
            database="connected" if db_healthy else "disconnected",
            ollama="connected" if ollama_healthy else "disconnected",
            knowledge_base={
                "documents": doc_count,
                "chunks": chunk_count
            },
            model_info={
                "llm_model": settings.ollama_llm_model,
                "embedding_model": settings.ollama_embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                "hybrid_search": settings.use_hybrid_search,
                "reranker_enabled": settings.reranker_enabled,
                "reranker_model": settings.reranker_model if settings.reranker_enabled else None
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        error_detail = {
            "error": "Health check failed",
            "message": str(e),
            "type": type(e).__name__,
            "hint": "Check if database and Ollama services are running"
        }
        raise HTTPException(status_code=503, detail=error_detail)


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Chat with the RAG assistant (non-streaming).
    
    The assistant searches the knowledge base and provides contextual answers.
    """
    try:
        # Convert message history to dict format
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        logger.info("Processing chat request", extra={
            "query_length": len(request.message),
            "has_history": bool(conversation_history)
        })
        
        # Generate response using RAG
        result = await rag_engine.chat(
            session,
            request.message,
            conversation_history,
            metadata_filter=request.metadata_filter or None,
        )
        
        # Convert back to ChatMessage format
        updated_history = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in result["conversation_history"]
        ]
        
        # Convert citations to Citation objects
        citations = [
            Citation(**citation)
            for citation in result.get("citations", [])
        ]
        
        logger.info("Chat request completed", extra={
            "response_length": len(result["response"]),
            "citations_count": len(citations)
        })
        
        # Fire-and-forget RAGAS background evaluation
        context_texts = [cit["content"] for cit in result.get("citations", []) if isinstance(cit, dict) and cit.get("content")]
        if context_texts and result["response"]:
            asyncio.create_task(
                background_ragas_evaluate(
                    question=request.message,
                    answer=result["response"],
                    contexts=context_texts,
                    session_factory=db_manager.get_session,
                )
            )
        
        return ChatResponse(
            response=result["response"],
            conversation_history=updated_history,
            citations=citations
        )
        
    except ValueError as e:
        if METRICS_AVAILABLE:
            metrics.rag_requests_total.labels(status="error").inc()
        logger.error(f"Chat validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid request",
                "message": str(e),
                "type": "ValueError"
            }
        )
    except ConnectionError as e:
        if METRICS_AVAILABLE:
            metrics.rag_requests_total.labels(status="error").inc()
        logger.error(f"Chat connection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Cannot connect to Ollama service",
                "details": str(e),
                "type": "ConnectionError",
                "hint": "Check if Ollama is running and accessible"
            }
        )
    except Exception as e:
        if METRICS_AVAILABLE:
            metrics.rag_requests_total.labels(status="error").inc()
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat processing failed",
                "message": str(e),
                "type": type(e).__name__,
                "hint": "Check logs for more details"
            }
        )


@router.post("/chat/stream", tags=["Chat"])
async def chat_stream(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Chat with the RAG assistant (streaming).
    
    Returns server-sent events with response chunks.
    """
    async def generate():
        try:
            # Convert message history
            conversation_history = None
            if request.conversation_history:
                conversation_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.conversation_history
                ]
            
            # Search knowledge base with full pipeline trace
            yield f"data: {json.dumps({'status': 'searching'})}\n\n"
            detailed = await rag_engine.search_detailed(
                session,
                request.message,
                metadata_filter=request.metadata_filter or None,
            )
            search_results = detailed["results"]
            
            # Emit retrieval trace (pre/post rerank, timing breakdown)
            yield f"data: {json.dumps({'status': 'retrieval_trace', 'trace': detailed['trace']})}\n\n"
            
            # Build citations
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
            
            # Send citations
            yield f"data: {json.dumps({'status': 'citations', 'citations': citations})}\n\n"
            
            # Stream response — pass pre-fetched results to skip a second search.
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"
            
            full_response = ""
            async for chunk in rag_engine.generate_answer_stream(
                session,
                request.message,
                conversation_history,
                search_results=search_results,
            ):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'done', 'response': full_response})}\n\n"
            
            # Record success metric
            if METRICS_AVAILABLE:
                metrics.rag_requests_total.labels(status="success").inc()
            
            # Fire-and-forget RAGAS evaluation in the background
            context_texts = [c["content"] for c in citations if c.get("content")]
            if context_texts and full_response:
                asyncio.create_task(
                    background_ragas_evaluate(
                        question=request.message,
                        answer=full_response,
                        contexts=context_texts,
                        session_factory=db_manager.get_session,
                    )
                )
            
        except Exception as e:
            if METRICS_AVAILABLE:
                metrics.rag_requests_total.labels(status="error").inc()
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# Search Endpoint
# ============================================================================

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    request: SearchRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Search the knowledge base using semantic similarity.
    
    Returns relevant chunks with similarity scores.
    """
    try:
        results = await rag_engine.search(
            session,
            request.query,
            request.limit,
            metadata_filter=request.metadata_filter or None,
        )
        
        items = [
            SearchResultItem(
                chunk_id=str(r.chunk_id),
                document_id=str(r.document_id),
                content=r.content,
                similarity=r.similarity,
                metadata=r.chunk_metadata,
                document_title=r.document_title,
                document_source=r.document_source
            )
            for r in results
        ]
        
        return SearchResponse(
            results=items,
            total_results=len(items)
        )
        
    except ValueError as e:
        logger.error(f"Search validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid search query",
                "message": str(e),
                "type": "ValueError"
            }
        )
    except ConnectionError as e:
        logger.error(f"Search connection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Embedding service unavailable",
                "message": "Cannot connect to Ollama for embeddings",
                "details": str(e),
                "type": "ConnectionError"
            }
        )
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Search failed",
                "message": str(e),
                "type": type(e).__name__,
                "hint": "Ensure knowledge base is populated and Ollama is running"
            }
        )


# ============================================================================
# Document Management Endpoints
# ============================================================================

@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_documents(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session)
):
    """
    List all documents in the knowledge base.
    
    Supports pagination with limit and offset parameters.
    """
    try:
        documents = await list_documents(session, limit=limit, offset=offset)
        total = await get_document_count(session)
        
        doc_infos = [
            DocumentInfo(
                id=str(doc.id),
                title=doc.title,
                source=doc.source,
                metadata=doc.metadata_,
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=doc_infos,
            total=total
        )
        
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to list documents",
                "message": str(e),
                "type": type(e).__name__,
                "hint": "Check database connection"
            }
        )


# ============================================================================
# Ingestion Endpoint
# ============================================================================

@router.post("/ingest", response_model=IngestionResponse, tags=["Documents"])
async def ingest_documents(
    request: IngestionRequest,
    session: AsyncSession = Depends(get_db_session)
):
    """
    Ingest documents from the documents folder.
    
    Processes all supported files (PDF, Word, Excel, PowerPoint, Markdown, Audio)
    and stores them in the knowledge base with embeddings.
    """
    try:
        # Run ingestion in background (non-blocking)
        pipeline = IngestionPipeline(
            documents_dir=request.documents_path,
            clean_existing=request.clean_existing
        )
        
        # Run the pipeline
        logger.info(f"Starting ingestion from {request.documents_path}")
        result = await asyncio.to_thread(pipeline.run)
        
        return IngestionResponse(
            status="completed" if result["success"] else "failed",
            message=result.get("message", "Ingestion completed"),
            documents_processed=result.get("documents_processed", 0),
            chunks_created=result.get("chunks_created", 0),
            errors=result.get("errors", [])
        )
        
    except FileNotFoundError as e:
        logger.error(f"Ingestion path error: {e}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Documents path not found",
                "message": str(e),
                "type": "FileNotFoundError",
                "hint": f"Check if path exists: {request.documents_path}"
            }
        )
    except PermissionError as e:
        logger.error(f"Ingestion permission error: {e}", exc_info=True)
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Permission denied",
                "message": str(e),
                "type": "PermissionError",
                "hint": "Check file/folder permissions"
            }
        )
    except Exception as e:
        logger.error(f"Ingestion endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Ingestion failed",
                "message": str(e),
                "type": type(e).__name__,
                "hint": "Check logs for detailed error information"
            }
        )


@router.post("/upload", response_model=FileUploadResponse, tags=["Documents"])
async def upload_file(
    file: UploadFile = File(..., description="File to upload (PDF, DOCX, PPTX, XLSX, MD, TXT, MP3, WAV, M4A, FLAC)")
):
    """
    Upload a single file and ingest it into the knowledge base.
    
    Supports: PDF, DOCX, PPTX, XLSX, MD, TXT, MP3, WAV, M4A, FLAC
    """
    # Validate file extension
    text_extensions = {
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
        '.md', '.txt', '.mp3', '.wav', '.m4a', '.flac'
    }
    image_extensions = {
        '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'
    }
    supported_extensions = text_extensions | (
        image_extensions if settings.image_captioning_enabled else set()
    )

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in supported_extensions:
        hint = (
            " Image files (.png, .jpg, etc.) require IMAGE_CAPTIONING_ENABLED=true."
            if file_ext in image_extensions
            else ""
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(sorted(supported_extensions))}.{hint}"
        )
    
    try:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename}")

        # ── Standalone image upload ──────────────────────────────────────────
        if file_ext in image_extensions:
            from backend.ingestion.image_extractor import ImageExtractor
            from backend.ingestion.image_captioner import get_captioner

            extractor = ImageExtractor(
                min_width=settings.image_min_width_px,
                min_height=settings.image_min_height_px,
            )
            extracted = extractor.extract(tmp_file_path)

            if not extracted:
                os.unlink(tmp_file_path)
                raise HTTPException(status_code=400, detail="Could not load image or image is too small")

            captioner = get_captioner(model_name=settings.blip_model)
            caption = captioner.caption(extracted[0].image)

            if not caption:
                os.unlink(tmp_file_path)
                raise HTTPException(status_code=400, detail="Image captioning returned empty caption")

            content_text = f"[Standalone image file: {file.filename}]: {caption}"
            title = os.path.splitext(file.filename)[0]

        # ── Text / structured document upload ────────────────────────────────
        else:
            # Initialize components
            config = ChunkingConfig(max_tokens=settings.max_tokens_per_chunk)
            chunker = DoclingHybridChunker(config)
            embedder = OllamaEmbedder()

            # Read document
            pipeline = IngestionPipeline(documents_folder="", clean_before_ingest=False)
            content_text, docling_doc = pipeline._read_document(tmp_file_path)
            title = pipeline._extract_title(content_text, tmp_file_path)
        
        # ── Chunk + embed (text path only) ───────────────────────────────────
        if file_ext not in image_extensions:
            # Chunk document
            chunks = await chunker.chunk_document(
                content=content_text,
                title=title,
                source=file.filename,
                metadata={"uploaded": True, "original_filename": file.filename},
                docling_doc=docling_doc
            )

            if not chunks:
                os.unlink(tmp_file_path)
                raise HTTPException(status_code=400, detail="No chunks could be created from the file")

            # Generate embeddings
            embedded_chunks = await embedder.embed_chunks(chunks)

        # ── Save to database ─────────────────────────────────────────────────
        async with db_manager.get_session() as session:
            document = await create_document(
                session,
                title=title or file.filename,
                source=file.filename,
                content=content_text,
                metadata={"uploaded": True, "original_filename": file.filename}
            )

            if file_ext in image_extensions:
                # Single image-caption chunk
                from backend.ingestion.embedder import OllamaEmbedder as _Emb
                from backend.ingestion.chunker import DocumentChunk as _DC
                _emb = _Emb()
                img_meta = extracted[0].to_metadata_dict()
                img_meta["uploaded"] = True
                img_meta["blip_model"] = settings.blip_model
                cap_chunk = _DC(
                    content=content_text,
                    index=0,
                    start_char=0,
                    end_char=len(content_text),
                    metadata=img_meta,
                )
                [ec] = await _emb.embed_chunks([cap_chunk])
                await create_chunk(
                    session,
                    document_id=document.id,
                    content=ec.content,
                    embedding=ec.embedding,
                    chunk_index=0,
                    metadata=ec.metadata,
                )
                chunks_saved = 1
            else:
                for chunk in embedded_chunks:
                    await create_chunk(
                        session,
                        document_id=document.id,
                        content=chunk.content,
                        embedding=chunk.embedding,
                        chunk_index=chunk.index,
                        metadata=chunk.metadata
                    )
                chunks_saved = len(chunks)

            await session.commit()
        
        # Clean up temp file
        os.unlink(tmp_file_path)

        logger.info(f"Successfully processed {file.filename}: {chunks_saved} chunk(s) created")

        return FileUploadResponse(
            status="success",
            message=f"File '{file.filename}' processed successfully",
            document_id=str(document.id),
            chunks_created=chunks_saved,
            filename=file.filename
        )
        
    except FileNotFoundError as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"File upload - file not found: {e}", exc_info=True)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "File processing failed",
                "message": "Temporary file was lost during processing",
                "details": str(e),
                "type": "FileNotFoundError"
            }
        )
    except ValueError as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"File upload - validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file content",
                "message": str(e),
                "type": "ValueError",
                "hint": "File may be corrupted or in an unsupported format"
            }
        )
    except ConnectionError as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"File upload - connection error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service unavailable",
                "message": "Cannot connect to Ollama for embeddings",
                "details": str(e),
                "type": "ConnectionError",
                "hint": "Ensure Ollama is running and accessible"
            }
        )
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        logger.error(f"File upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "File processing failed",
                "message": str(e),
                "type": type(e).__name__,
                "filename": file.filename,
                "hint": "Check if file is valid and Ollama service is running"
            }
        )


# ============================================================================
# RAGAS Evaluation Endpoints
# ============================================================================

def _record_to_response(record) -> RAGASEvaluationResponse:
    """Convert a RAGASEvaluation DB record to the API response schema."""
    return RAGASEvaluationResponse(
        id=str(record.id),
        question=record.question,
        answer=record.answer,
        scores=RAGASScores(
            faithfulness=record.faithfulness,
            answer_relevancy=record.answer_relevancy,
            context_precision=record.context_precision,
            context_recall=record.context_recall,
        ),
        model_used=record.model_used,
        evaluated_at=record.evaluated_at,
        has_reference=record.reference is not None,
    )


def _compute_averages(evals: list[RAGASEvaluationResponse]) -> RAGASScores:
    """Compute rolling averages over all evaluations, ignoring None values."""
    def _avg(values):
        real = [v for v in values if v is not None]
        return sum(real) / len(real) if real else None

    return RAGASScores(
        faithfulness=_avg([e.scores.faithfulness for e in evals]),
        answer_relevancy=_avg([e.scores.answer_relevancy for e in evals]),
        context_precision=_avg([e.scores.context_precision for e in evals]),
        context_recall=_avg([e.scores.context_recall for e in evals]),
    )


@router.post(
    "/evaluate",
    response_model=RAGASEvaluationResponse,
    tags=["Evaluation"],
    summary="Run RAGAS evaluation on a RAG output triple",
)
async def evaluate_rag(
    request: RAGASEvaluationRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """
    Evaluate a RAG question/answer/context triple with RAGAS metrics.

    - **Faithfulness** and **AnswerRelevancy** are always computed.
    - **ContextPrecision** and **ContextRecall** are only computed when
      `reference` (ground-truth answer) is provided.

    The evaluation runs Ollama locally as the judge LLM.
    Expect 15-60 seconds depending on hardware.
    """
    from backend.core.ragas_evaluator import run_ragas_evaluation
    from backend.database.operations import save_ragas_evaluation

    scores = await run_ragas_evaluation(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        reference=request.reference,
    )
    record = await save_ragas_evaluation(
        session=session,
        question=request.question,
        answer=request.answer,
        contexts=request.contexts,
        scores=scores,
        reference=request.reference,
    )
    return _record_to_response(record)


@router.get(
    "/evaluate/history",
    response_model=RAGASHistoryResponse,
    tags=["Evaluation"],
    summary="Fetch recent RAGAS evaluations with rolling averages",
)
async def get_evaluation_history(
    limit: int = 50,
    session: AsyncSession = Depends(get_db_session),
):
    """Return the most recent RAGAS evaluations and rolling metric averages."""
    from backend.database.operations import get_ragas_history

    records = await get_ragas_history(session, limit=limit)
    evals = [_record_to_response(r) for r in records]
    return RAGASHistoryResponse(
        evaluations=evals,
        total=len(evals),
        averages=_compute_averages(evals),
    )


async def background_ragas_evaluate(
    question: str,
    answer: str,
    contexts: list[str],
    session_factory,
) -> None:
    """
    Fire-and-forget task: evaluate a chat turn with RAGAS and store the result.

    Called via asyncio.create_task() so it never blocks the SSE stream.
    Opens its own DB session because the originating request session will
    have closed by the time evaluation finishes.
    """
    from backend.core.ragas_evaluator import run_ragas_evaluation
    from backend.database.operations import save_ragas_evaluation

    try:
        scores = await run_ragas_evaluation(
            question=question,
            answer=answer,
            contexts=contexts,
        )
        async with session_factory() as session:
            await save_ragas_evaluation(
                session=session,
                question=question,
                answer=answer,
                contexts=contexts,
                scores=scores,
            )
        logger.info(
            f"RAGAS background evaluation saved for question={question[:60]!r}"
        )
    except Exception as exc:
        logger.error(f"RAGAS background evaluation failed: {exc}", exc_info=True)
