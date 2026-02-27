"""
Document ingestion pipeline for the RAG system.
Processes documents, chunks them, generates embeddings, and stores in database.
"""

import argparse
import asyncio
import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from docling.document_converter import DocumentConverter, AudioFormatOption
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import InputFormat
from docling.pipeline.asr_pipeline import AsrPipeline

from backend.config import settings
from backend.database.connection import db_manager, get_db_session
from backend.database.operations import (
    create_document,
    create_chunk,
    delete_all_documents,
    get_document_count,
    get_chunk_count
)
from backend.ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from backend.ingestion.embedder import OllamaEmbedder

try:
    from backend.core.observability import metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting documents into the RAG system."""
    
    def __init__(
        self,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_semantic_chunking: bool = True
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            documents_folder: Folder containing documents
            clean_before_ingest: Whether to clean existing data first
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            use_semantic_chunking: Use Docling HybridChunker
        """
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        
        # Initialize chunker
        chunker_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_tokens=settings.max_tokens_per_chunk,
            use_semantic_splitting=use_semantic_chunking
        )
        self.chunker = create_chunker(chunker_config)
        
        # Initialize embedder
        self.embedder = OllamaEmbedder()
    
    def _find_document_files(self) -> List[str]:
        """Find all supported document files."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []
        
        # Supported file patterns
        patterns = [
            "*.md", "*.markdown", "*.txt", 
            "*.pdf",  
            "*.docx", "*.doc", 
            "*.pptx", "*.ppt", 
            "*.xlsx", "*.xls", 
            "*.html", "*.htm",  
            "*.mp3", "*.wav", "*.m4a", "*.flac", 
        ]
        
        files = []
        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(self.documents_folder, "**", pattern),
                    recursive=True
                )
            )
        
        return sorted(files)
    
    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Read document content from file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (content, docling_document)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Audio formats - transcribe with Whisper
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac']
        if file_ext in audio_formats:
            content = self._transcribe_audio(file_path)
            return (content, None)
        
        # Docling-supported formats
        docling_formats = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm']
        
        if file_ext in docling_formats:
            try:
                logger.info(f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}")
                converter = DocumentConverter()
                result = converter.convert(file_path)
                markdown_content = result.document.export_to_markdown()
                logger.info(f"Successfully converted {os.path.basename(file_path)}")
                
                return (markdown_content, result.document)
                
            except Exception as e:
                logger.error(f"Failed to convert {file_path} with Docling: {e}")
                # Fall back to raw text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return (f.read(), None)
                except:
                    return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)
        
        # Text-based formats
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return (f.read(), None)
    
    def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio file using Whisper via Docling."""
        try:
            audio_path = Path(file_path).resolve()
            logger.info(f"Transcribing audio: {audio_path.name}")
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            
            result = converter.convert(audio_path)
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to transcribe {file_path}: {e}")
            return f"[Error: Could not transcribe {os.path.basename(file_path)}]"
    
    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        # Try to find markdown title
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback to filename
        return os.path.splitext(os.path.basename(file_path))[0]
    
    async def _ingest_single_document(
        self,
        session: AsyncSession,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Ingest a single document.
        
        Args:
            session: Database session
            file_path: Path to document file
            
        Returns:
            Ingestion result dictionary
        """
        start_time = datetime.now()
        
        # Read document
        content, docling_doc = self._read_document(file_path)
        title = self._extract_title(content, file_path)
        source = os.path.relpath(file_path, self.documents_folder)
        
        logger.info(f"Processing: {title}")
        
        # Chunk document
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata={"file_path": file_path},
            docling_doc=docling_doc
        )
        
        if not chunks:
            logger.warning(f"No chunks created for {title}")
            return {
                "title": title,
                "chunks_created": 0,
                "success": False,
                "error": "No chunks created"
            }
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Save to database
        document = await create_document(
            session,
            title=title,
            source=source,
            content=content,
            metadata={"file_path": file_path, "ingestion_date": datetime.now().isoformat()}
        )
        
        for chunk in embedded_chunks:
            await create_chunk(
                session,
                document_id=document.id,
                content=chunk.content,
                embedding=chunk.embedding,
                chunk_index=chunk.index,
                token_count=chunk.token_count,
                metadata=chunk.metadata
            )
        
        await session.commit()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Saved document to database: {document.id}")
        
        # Record metrics
        if METRICS_AVAILABLE:
            metrics.ingestion_documents_total.labels(status="success").inc()
            metrics.ingestion_chunks_created.inc(len(chunks))
            metrics.ingestion_duration.observe(processing_time)
        
        return {
            "title": title,
            "chunks_created": len(chunks),
            "processing_time": processing_time,
            "success": True
        }
    
    async def run(self):
        """Run the ingestion pipeline."""
        logger.info("Starting document ingestion pipeline...")
        
        # Initialize database
        if not db_manager.engine:
            await db_manager.initialize()
        
        # Clean database if requested
        if self.clean_before_ingest:
            logger.warning("Cleaning existing data...")
            async with db_manager.get_session() as session:
                deleted = await delete_all_documents(session)
                await session.commit()
                logger.info(f"Deleted {deleted} existing documents")
        
        # Find documents
        document_files = self._find_document_files()
        
        if not document_files:
            logger.warning(f"No documents found in {self.documents_folder}")
            return
        
        logger.info(f"Found {len(document_files)} documents to process")
        
        # Process documents
        results = []
        for i, file_path in enumerate(document_files, 1):
            logger.info(f"\n[{i}/{len(document_files)}] Processing: {os.path.basename(file_path)}")
            
            try:
                async with db_manager.get_session() as session:
                    result = await self._ingest_single_document(session, file_path)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
                results.append({
                    "title": os.path.basename(file_path),
                    "chunks_created": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        
        successful = sum(1 for r in results if r["success"])
        total_chunks = sum(r["chunks_created"] for r in results)
        errors = [r.get("error", "Unknown error") for r in results if not r["success"]]
        
        print(f"Documents processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Total chunks created: {total_chunks}")
        print()
        
        # Print individual results
        for result in results:
            status = "✓" if result["success"] else "✗"
            print(f"{status} {result['title']}: {result['chunks_created']} chunks")
            if not result["success"]:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("="*60)
        
        # Final stats
        async with db_manager.get_session() as session:
            doc_count = await get_document_count(session)
            chunk_count = await get_chunk_count(session)
            print(f"\nKnowledge base now contains:")
            print(f"  Documents: {doc_count}")
            print(f"  Chunks: {chunk_count}")
        
        # Return results for API
        return {
            "success": successful > 0,
            "message": f"Processed {len(results)} documents successfully" if successful > 0 else "All documents failed to process",
            "documents_processed": successful,
            "chunks_created": total_chunks,
            "errors": errors
        }


async def main():
    """Main entry point for ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument(
        "--documents", "-d",
        default="documents",
        help="Documents folder path"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't clean existing data before ingestion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap size"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic chunking"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and run pipeline
    pipeline = IngestionPipeline(
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic
    )
    
    try:
        await pipeline.run()
    except KeyboardInterrupt:
        logger.info("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
