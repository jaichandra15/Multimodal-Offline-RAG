"""
Database connection and session management using async SQLAlchemy.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from sqlalchemy.pool import NullPool

from backend.config import settings
from backend.database.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        """Initialize database manager."""
        self.engine: AsyncEngine | None = None
        self.async_session_factory: async_sessionmaker[AsyncSession] | None = None
    
    async def initialize(self):
        """Initialize database engine and session factory."""
        if self.engine is not None:
            logger.warning("Database already initialized")
            return
        
        logger.info(f"Initializing database connection...")
        
        # Create async engine
        self.engine = create_async_engine(
            settings.database_url.replace('postgresql://', 'postgresql+asyncpg://'),
            echo=settings.log_level == "DEBUG",
            pool_size=settings.db_pool_min_size,
            max_overflow=settings.db_pool_max_size - settings.db_pool_min_size,
            pool_pre_ping=True,
        )
        
        # Create session factory
        self.async_session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        logger.info("Database connection initialized successfully")
    
    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
            self.engine = None
            self.async_session_factory = None
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session.
        
        Yields:
            AsyncSession instance
        """
        if self.async_session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def create_tables(self):
        """Create all tables in the database."""
        if self.engine is None:
            raise RuntimeError("Database not initialized")
        
        async with self.engine.begin() as conn:
            # Enable pgvector extension (must wrap in text())
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            # Enable pg_trgm extension for fuzzy text matching
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Create HNSW index for faster vector search
            # HNSW (Hierarchical Navigable Small World) provides much faster approximate nearest neighbor search
            # m=16: number of connections per layer (higher = more accurate but slower build)
            # ef_construction=64: size of dynamic candidate list during construction
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx 
                ON chunks USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            logger.info("HNSW index created for vector search")
            
            # Create GIN index for full-text search (BM25-style keyword search)
            # This enables hybrid search combining vector + keyword matching
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS chunks_content_fts_idx 
                ON chunks USING gin (to_tsvector('english', content))
            """))
            logger.info("GIN index created for full-text search")
            
            # Create GIN index for trigram fuzzy matching
            # This enables fuzzy keyword search that handles OCR errors and typos
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx 
                ON chunks USING gin (content gin_trgm_ops)
            """))
            logger.info("Trigram index created for fuzzy search")
        
        logger.info("Database tables created")
    
    async def drop_tables(self):
        """Drop all tables in the database."""
        if self.engine is None:
            raise RuntimeError("Database not initialized")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Database tables dropped")
    
    async def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        AsyncSession instance
    """
    async with db_manager.get_session() as session:
        yield session
