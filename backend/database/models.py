"""
SQLAlchemy models for documents and chunks with pgvector support.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from backend.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class Document(Base):
    """Document model - stores original documents with metadata."""
    
    __tablename__ = "documents"
    
    id: Mapped[UUID] = mapped_column(
        PostgreSQL_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    
    # Relationship to chunks
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title}')>"


class Chunk(Base):
    """Chunk model - stores document chunks with embeddings."""
    
    __tablename__ = "chunks"
    
    id: Mapped[UUID] = mapped_column(
        PostgreSQL_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQL_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[Vector]] = mapped_column(
        Vector(settings.embedding_dimensions),
        nullable=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False
    )
    
    # Relationship to document
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )
    
    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class RAGASEvaluation(Base):
    """
    Stores RAGAS evaluation results for individual RAG interactions.
    Scores are nullable floats — None means the metric was not computed.
    """

    __tablename__ = "ragas_evaluations"

    id: Mapped[UUID] = mapped_column(
        PostgreSQL_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    # The user question that was evaluated
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # The RAG-generated answer
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON list[str] of retrieved chunk texts used as context
    contexts: Mapped[list] = mapped_column(JSON, nullable=False)
    # Optional ground-truth reference answer (enables Context Precision / Recall)
    reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # RAGAS scores (nullable = not computed)
    faithfulness: Mapped[Optional[float]] = mapped_column(
        "faithfulness", nullable=True
    )
    answer_relevancy: Mapped[Optional[float]] = mapped_column(
        "answer_relevancy", nullable=True
    )
    context_precision: Mapped[Optional[float]] = mapped_column(
        "context_precision", nullable=True
    )
    context_recall: Mapped[Optional[float]] = mapped_column(
        "context_recall", nullable=True
    )

    model_used: Mapped[str] = mapped_column(String, nullable=False, default="mistral")
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<RAGASEvaluation(id={self.id}, "
            f"faithfulness={self.faithfulness:.3f if self.faithfulness else 'N/A'}, "
            f"answer_relevancy={self.answer_relevancy:.3f if self.answer_relevancy else 'N/A'})>"
        )
