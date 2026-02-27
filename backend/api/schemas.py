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
