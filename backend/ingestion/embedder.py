"""
Document embedding generation using Ollama.
"""

import logging
from typing import List

from backend.core.ollama_client import ollama_client
from backend.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Generates embeddings for document chunks using Ollama."""
    
    def __init__(self, batch_size: int = 50):
        """
        Initialize embedder.
        
        Args:
            batch_size: Number of chunks to process at a time
        """
        self.ollama = ollama_client
        self.batch_size = batch_size
    
    async def embed_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Generate embedding for a single chunk.
        
        Args:
            chunk: Document chunk to embed
            
        Returns:
            Chunk with embedding added
        """
        embedding = await self.ollama.generate_embedding(chunk.content)
        chunk.embedding = embedding
        return chunk
    
    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback=None
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of document chunks
            progress_callback: Optional callback for progress updates
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            current_batch = (i // self.batch_size) + 1
            
            logger.info(f"Processing batch {current_batch}/{total_batches}...")
            
            for chunk in batch_chunks:
                try:
                    embedded_chunk = await self.embed_chunk(chunk)
                    embedded_chunks.append(embedded_chunk)
                except Exception as e:
                    logger.error(f"Failed to embed chunk {chunk.index}: {e}")
                    # Add chunk without embedding
                    embedded_chunks.append(chunk)
            
            if progress_callback:
                progress_callback(current_batch, total_batches)
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
