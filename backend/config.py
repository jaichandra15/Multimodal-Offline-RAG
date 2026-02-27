"""
Configuration settings for the RAG backend.
All settings are loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic Settings v2."""
    
    # Database Configuration
    database_url: str
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_command_timeout: int = 60
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "mistral"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_timeout: int = 300  
    
    # Embedding Configuration
    embedding_dimensions: int = 768  
    max_tokens_per_chunk: int = 512
    
    # RAG Configuration
    top_k_results: int = 5
    similarity_threshold: float = 0.3
    use_hybrid_search: bool = True  # Enable hybrid search (vector + keyword)
    hybrid_vector_weight: float = 0.6  # Weight for vector search in hybrid mode
    hybrid_keyword_weight: float = 0.4  # Weight for keyword search in hybrid mode
    
    # Reranker Configuration
    reranker_enabled: bool = True  # Enable cross-encoder reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 30  # Number of candidates to fetch for reranking
    reranker_batch_size: int = 32  # Batch size for reranker inference
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "RAG Knowledge Assistant API"
    api_version: str = "2.0.0"
    cors_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
