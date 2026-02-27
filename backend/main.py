"""
Main FastAPI application for RAG backend.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.config import settings
from backend.database.connection import db_manager
from backend.api.routes import router
from backend.core.observability import (
    configure_logging,
    setup_metrics,
    RequestIDMiddleware
)

# Configure structured logging
configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting RAG Backend API...")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"LLM Model: {settings.ollama_llm_model}")
    logger.info(f"Embedding Model: {settings.ollama_embedding_model}")
    
    try:
        # Initialize DB connection
        await db_manager.initialize()
        logger.info("Database initialized successfully")

        # Ensure tables exist
        logger.info("Creating database tables if not present...")
        await db_manager.create_tables()

        # Health check
        if await db_manager.health_check():
            logger.info("Database connection verified")
        else:
            logger.warning("Database health check failed")

        yield

    finally:
        logger.info("Shutting down RAG Backend API...")
        await db_manager.close()
        logger.info("Database connections closed")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="RAG Knowledge Assistant API powered by Ollama",
    lifespan=lifespan
)

# Add observability middleware
app.add_middleware(RequestIDMiddleware)

# Setup Prometheus metrics
setup_metrics(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Global Exception Handlers
# ============================================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with detailed error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail},
            "path": str(request.url),
            "method": request.method
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed field information."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": errors,
            "path": str(request.url),
            "method": request.method
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url),
            "method": request.method,
            "hint": "Check server logs for detailed error information"
        }
    )


# ============================================================================
# Routes
# ============================================================================

app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Knowledge Assistant API",
        "version": settings.api_version,
        "llm_model": settings.ollama_llm_model,
        "embedding_model": settings.ollama_embedding_model,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
