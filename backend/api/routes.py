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
    FileUploadResponse
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
        result = await rag_engine.chat(session, request.message, conversation_history)
        
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
            
            # Search knowledge base for citations
            yield f"data: {json.dumps({'status': 'searching'})}\n\n"
            search_results = await rag_engine.search(session, request.message)
            
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
            
            # Stream response
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"
            
            full_response = ""
            async for chunk in rag_engine.generate_answer_stream(
                session,
                request.message,
                conversation_history
            ):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'done', 'response': full_response})}\n\n"
            
            # Record success metric
            if METRICS_AVAILABLE:
                metrics.rag_requests_total.labels(status="success").inc()
            
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
        results = await rag_engine.search(session, request.query, request.limit)
        
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
    supported_extensions = {
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
        '.md', '.txt', '.mp3', '.wav', '.m4a', '.flac'
    }
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(supported_extensions)}"
        )
    
    try:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Initialize components
        config = ChunkingConfig(max_tokens=settings.max_tokens_per_chunk)
        chunker = DoclingHybridChunker(config)
        embedder = OllamaEmbedder()
        
        # Read document
        pipeline = IngestionPipeline(documents_folder="", clean_before_ingest=False)
        content_text, docling_doc = pipeline._read_document(tmp_file_path)
        title = pipeline._extract_title(content_text, tmp_file_path)
        
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
        
        # Save to database
        async with db_manager.get_session() as session:
            document = await create_document(
                session,
                title=title or file.filename,
                source=file.filename,
                content=content_text,
                metadata={"uploaded": True, "original_filename": file.filename}
            )
            
            for chunk in embedded_chunks:
                await create_chunk(
                    session,
                    document_id=document.id,
                    content=chunk.content,
                    embedding=chunk.embedding,
                    chunk_index=chunk.index,
                    metadata=chunk.metadata
                )
            
            await session.commit()
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks created")
        
        return FileUploadResponse(
            status="success",
            message=f"File '{file.filename}' processed successfully",
            document_id=str(document.id),
            chunks_created=len(chunks),
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
