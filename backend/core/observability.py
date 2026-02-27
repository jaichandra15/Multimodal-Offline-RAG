"""
Observability module for RAG Backend.
Includes structured logging, metrics, and request tracing.
"""

import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from backend.config import settings

# Context variable for request ID
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter to include request ID and other context."""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
            
        # Add log level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
            
        # Add request ID if available
        req_id = request_id_ctx.get()
        if req_id:
            log_record['request_id'] = req_id
            
        # Add source location
        log_record['source'] = f"{record.name}:{record.lineno}"

def configure_logging():
    """Configure structured JSON logging."""
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    log_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers = [log_handler]
    root_logger.setLevel(settings.log_level)
    
    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to context and response headers."""
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request_id_ctx.set(request_id)
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

# Metrics definitions
class Metrics:
    """Prometheus metrics for the application."""
    
    # RAG Engine Metrics
    rag_search_latency = Histogram(
        "rag_search_latency_seconds",
        "Latency of RAG search operations",
        ["method"]  # vector, keyword, hybrid
    )
    
    rag_generation_latency = Histogram(
        "rag_generation_latency_seconds",
        "Latency of LLM generation",
        ["model"]
    )
    
    rag_chunks_retrieved = Histogram(
        "rag_chunks_retrieved_count",
        "Number of chunks retrieved per search",
        buckets=[0, 1, 3, 5, 10, 20, 50]
    )
    
    rag_requests_total = Counter(
        "rag_requests_total",
        "Total number of RAG requests",
        ["status"]  # success, error
    )
    
    # Ingestion Metrics
    ingestion_documents_total = Counter(
        "ingestion_documents_total",
        "Total documents processed",
        ["status"]
    )
    
    ingestion_chunks_created = Counter(
        "ingestion_chunks_created_total",
        "Total chunks created"
    )
    
    ingestion_duration = Histogram(
        "ingestion_duration_seconds",
        "Time taken to ingest a document"
    )
    
    # Reranker Metrics
    reranker_latency = Histogram(
        "reranker_latency_seconds",
        "Latency of reranking operations",
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    reranker_rank_change = Histogram(
        "reranker_rank_change_positions",
        "How many positions the top result moved after reranking",
        buckets=[0, 1, 2, 3, 5, 10, 20, 50]
    )
    
    reranker_calls_total = Counter(
        "reranker_calls_total",
        "Total number of reranker calls",
        ["status"]  # success, error, disabled
    )

metrics = Metrics()

def setup_metrics(app):
    """Initialize Prometheus metrics for FastAPI app."""
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
    )
    instrumentator.instrument(app).expose(app)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured formatting."""
    return logging.getLogger(name)
