"""
Docling HybridChunker implementation for intelligent document splitting.

This module uses Docling's built-in HybridChunker which combines:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)

Benefits over custom chunking:
- Fast (no LLM API calls)
- Token-precise (not character-based estimates)
- Better for RAG (chunks include document context)
- Battle-tested (maintained by Docling team)
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    max_chunk_size: int = 2000  # Maximum chunk size
    min_chunk_size: int = 100  # Minimum chunk size
    use_semantic_splitting: bool = True 
    preserve_structure: bool = True  # Preserve document structure
    max_tokens: int = 512  # Maximum tokens for embedding models

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """Represents a document chunk with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None  

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            self.token_count = len(self.content) // 4


class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.

    This chunker uses Docling's built-in HybridChunker which:
    - Respects document structure (sections, paragraphs, tables)
    - Is token-aware (fits embedding model limits)
    - Preserves semantic coherence
    - Includes heading context in chunks
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config

        # Initialize tokenizer for token-aware chunking
        # Using bert-base-uncased as a generic tokenizer that works well with most embedding models
        # This avoids mismatch issues when using different embedding models like nomic-embed-text
        model_id = "bert-base-uncased"
        logger.info(f"Initializing tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Create HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )

        logger.info(f"HybridChunker initialized (max_tokens={config.max_tokens})")

    def _build_contextual_prefix(self, title: str, source: str, chunk_index: int, total_chunks: int) -> str:
        """
        Build a contextual prefix for a chunk to improve retrieval.
        
        This prefix helps embedding models understand:
        - Which document the chunk belongs to
        - The source/location of the document
        - The chunk's position in the document
        
        Args:
            title: Document title
            source: Document source path
            chunk_index: Index of this chunk (0-based)
            total_chunks: Total number of chunks in document
            
        Returns:
            Contextual prefix string
        """
        return f"Document: {title}\nSource: {source}\nChunk: {chunk_index + 1}/{total_chunks}"

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format)
            title: Document title
            source: Document source
            metadata: Additional metadata
            docling_doc: Optional pre-converted DoclingDocument (for efficiency)

        Returns:
            List of document chunks with contextualized content
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {})
        }
        if docling_doc is None:
            logger.warning("No DoclingDocument provided, using simple chunking fallback")
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                contextualized_text = self.chunker.contextualize(chunk=chunk)
                
                # Add contextual prefix for better retrieval
                # This helps the embedding model understand the context of the chunk
                contextual_prefix = self._build_contextual_prefix(title, source, i, len(chunks))
                prefixed_content = f"{contextual_prefix}\n\n{contextualized_text.strip()}"

                # Count actual tokens
                token_count = len(self.tokenizer.encode(prefixed_content))

                # Extract page information from chunk metadata
                page_number = None
                if hasattr(chunk, 'meta') and chunk.meta:
                    # Try to get page number from chunk metadata
                    if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                        # Get the first doc item's page number
                        first_item = chunk.meta.doc_items[0]
                        if hasattr(first_item, 'prov') and first_item.prov:
                            for prov in first_item.prov:
                                if hasattr(prov, 'page_no'):
                                    page_number = prov.page_no
                                    break

                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True,
                    "has_prefix": True
                }
                
                # Add page number if available
                if page_number is not None:
                    chunk_metadata["page"] = page_number

                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(prefixed_content)

                document_chunks.append(DocumentChunk(
                    content=prefixed_content,
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata,
                    token_count=token_count
                ))

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Simple fallback chunking when HybridChunker can't be used.

        This is used when:
        - No DoclingDocument is provided
        - HybridChunker fails

        Args:
            content: Content to chunk
            base_metadata: Base metadata for chunks

        Returns:
            List of document chunks
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple sliding window approach
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunk_text = content[start:]
            else:
                chunk_end = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < len(content) and content[i] in '.!?\n':
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))

                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1  
                    },
                    token_count=token_count
                ))

                chunk_index += 1
            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


class SimpleChunker:
    """
    Simple non-semantic chunker for faster processing without Docling.

    This is kept as a lightweight alternative when:
    - Speed is critical
    - Document structure is simple
    - Token precision is not required
    """

    def __init__(self, config: ChunkingConfig):
        """Initialize simple chunker."""
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs  
    ) -> List[DocumentChunk]:
        """
        Chunk document using simple paragraph-based rules.

        Args:
            content: Document content
            title: Document title
            source: Document source
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple",
            **(metadata or {})
        }

        # Split on double newlines (paragraphs)
        import re
        paragraphs = re.split(r'\n\s*\n', content)

        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph exceeds chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        chunk_index,
                        current_pos,
                        current_pos + len(current_chunk),
                        base_metadata.copy()
                    ))

                    current_pos += len(current_chunk)
                    chunk_index += 1

                # Start new chunk with current paragraph
                current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                chunk_index,
                current_pos,
                current_pos + len(current_chunk),
                base_metadata.copy()
            ))

        # Update total chunks in metadata
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        start_pos: int,
        end_pos: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        return DocumentChunk(
            content=content.strip(),
            index=index,
            start_char=start_pos,
            end_char=end_pos,
            metadata=metadata
        )


# Factory function
def create_chunker(config: ChunkingConfig):
    """
    Create appropriate chunker based on configuration.

    Args:
        config: Chunking configuration

    Returns:
        Chunker instance
    """
    if config.use_semantic_splitting:
        return DoclingHybridChunker(config)
    else:
        return SimpleChunker(config)
