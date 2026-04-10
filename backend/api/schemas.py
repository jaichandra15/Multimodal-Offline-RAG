"""
Pydantic schemas for API request and response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ============================================================================
# Chat Schemas
# ============================================================================

class Citation(BaseModel):
    """Source citation for a response."""
    number: int = Field(..., description="Citation number in the response")
    chunk_id: str = Field(..., description="ID of the source chunk")
    document_id: str = Field(..., description="ID of the source document")
    document_title: str = Field(..., description="Title of the source document")
    document_source: str = Field(..., description="Source path/filename")
    content: str = Field(..., description="Content of the cited chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (page, timestamp, etc.)")
    similarity: Optional[float] = Field(default=None, description="Similarity score if from search")


class ChatMessage(BaseModel):
    """Message in conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    message: str = Field(..., description="User's question or message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional conversation history for context"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSONB metadata filter applied to chunk metadata during retrieval. "
            "Only chunks whose metadata contains ALL specified key-value pairs "
            "will be returned. Example: {\"file_path\": \"report.pdf\"} or "
            "{\"page\": 5}"
        )
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str = Field(..., description="Assistant's response")
    conversation_history: List[ChatMessage] = Field(
        ...,
        description="Updated conversation history"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Source citations for the response"
    )


# ============================================================================
# Search Schemas
# ============================================================================

class SearchRequest(BaseModel):
    """Request for search endpoint."""
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSONB metadata filter applied to chunk metadata. "
            "Only chunks whose metadata contains ALL specified key-value pairs "
            "will be returned. Example: {\"page\": 3} or "
            "{\"file_path\": \"report.pdf\"}"
        )
    )


class SearchResultItem(BaseModel):
    """Single search result."""
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]
    document_title: str
    document_source: str


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    results: List[SearchResultItem]
    total_results: int


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentInfo(BaseModel):
    """Document information."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response for list documents endpoint."""
    documents: List[DocumentInfo]
    total: int


# ============================================================================
# Health Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    ollama: str
    knowledge_base: Dict[str, int]
    model_info: Dict[str, Any]


# ============================================================================
# Ingestion Schemas
# ============================================================================

class IngestionRequest(BaseModel):
    """Request for document ingestion."""
    clean_existing: Optional[bool] = Field(
        default=False,
        description="Whether to clean existing documents before ingestion"
    )
    documents_path: Optional[str] = Field(
        default="documents",
        description="Path to documents folder (relative to app root)"
    )


class IngestionResponse(BaseModel):
    """Response from ingestion endpoint."""
    status: str
    message: str
    documents_processed: int
    chunks_created: int
    errors: List[str] = []


class FileUploadResponse(BaseModel):
    """Response from file upload endpoint."""
    status: str
    message: str
    document_id: Optional[str] = None
    chunks_created: int
    filename: str


# ============================================================================
# RAGAS Evaluation Schemas
# ============================================================================

class RAGASEvaluationRequest(BaseModel):
    """Request to run RAGAS evaluation on a RAG output triple."""
    question: str = Field(..., description="The user question")
    answer: str = Field(..., description="The RAG-generated answer")
    contexts: List[str] = Field(
        ...,
        description="List of retrieved chunk texts used as context",
        min_length=1,
    )
    reference: Optional[str] = Field(
        default=None,
        description=(
            "Optional ground-truth answer. When provided, also computes "
            "ContextPrecision and ContextRecall."
        ),
    )


class RAGASScores(BaseModel):
    """RAGAS evaluation scores. None = metric not computed."""
    faithfulness: Optional[float] = Field(
        default=None,
        description="0-1. Is the answer grounded in the retrieved context?",
    )
    answer_relevancy: Optional[float] = Field(
        default=None,
        description="0-1. Does the answer address the question?",
    )
    context_precision: Optional[float] = Field(
        default=None,
        description="0-1. Fraction of retrieved context that is relevant (needs reference).",
    )
    context_recall: Optional[float] = Field(
        default=None,
        description="0-1. Fraction of required info that was retrieved (needs reference).",
    )


class RAGASEvaluationResponse(BaseModel):
    """Single RAGAS evaluation result."""
    id: str
    question: str
    answer: str
    scores: RAGASScores
    model_used: str
    evaluated_at: datetime
    has_reference: bool = False


class RAGASHistoryResponse(BaseModel):
    """RAGAS evaluation history with rolling averages."""
    evaluations: List[RAGASEvaluationResponse]
    total: int
    averages: RAGASScores

