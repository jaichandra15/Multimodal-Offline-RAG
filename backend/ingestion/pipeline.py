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
from backend.ingestion.image_extractor import ImageExtractor

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

        # Initialize image extractor (always available; lightweight)
        self.image_extractor = ImageExtractor(
            min_width=settings.image_min_width_px,
            min_height=settings.image_min_height_px,
            max_images_per_page=settings.image_max_per_page,
        )

        # BLIP captioner is loaded lazily only when captioning is enabled
        self._captioner = None
    
    def _find_document_files(self) -> List[str]:
        """Find all supported document files (text/audio + standalone images)."""
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

        # Include standalone image files when captioning is enabled
        if settings.image_captioning_enabled:
            patterns += [
                "*.png", "*.jpg", "*.jpeg",
                "*.webp", "*.gif", "*.bmp", "*.tiff",
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
    
    def _get_captioner(self):
        """Lazy-load the BLIP captioner (downloads model on first call)."""
        if self._captioner is None:
            from backend.ingestion.image_captioner import get_captioner
            self._captioner = get_captioner(model_name=settings.blip_model)
        return self._captioner

    async def _ingest_image_chunks(
        self,
        session: AsyncSession,
        document_id,
        file_path: str,
        base_chunk_index: int,
    ) -> int:
        """
        Extract images from *file_path*, caption each one with BLIP, embed
        the caption, and store it as a Chunk in the database.

        Returns the number of image-caption chunks created.
        """
        if not settings.image_captioning_enabled:
            return 0

        extracted = self.image_extractor.extract(file_path)
        if not extracted:
            logger.debug(f"No images extracted from {os.path.basename(file_path)}")
            return 0

        captioner = self._get_captioner()
        created = 0

        for img_data in extracted:
            try:
                caption = captioner.caption(img_data.image)
                if not caption:
                    logger.warning(
                        f"Empty caption for image #{img_data.image_index} "
                        f"on page {img_data.page_or_slide} of {os.path.basename(file_path)}"
                    )
                    continue

                # Format caption as a content string that retrieval can match on
                location = (
                    f"slide {img_data.page_or_slide}"
                    if file_path.lower().endswith((".pptx", ".ppt"))
                    else f"page {img_data.page_or_slide}" if img_data.page_or_slide > 0
                    else "standalone"
                )
                content = f"[Image on {location}]: {caption}"

                # Build chunk metadata
                chunk_meta = {
                    **img_data.to_metadata_dict(),
                    "blip_model": settings.blip_model,
                }

                # Generate text embedding for the caption
                caption_chunk = DocumentChunk(
                    content=content,
                    index=base_chunk_index + created,
                    start_char=0,
                    end_char=len(content),
                    metadata=chunk_meta,
                )
                embedded_chunks = await self.embedder.embed_chunks([caption_chunk])

                for ec in embedded_chunks:
                    await create_chunk(
                        session,
                        document_id=document_id,
                        content=ec.content,
                        embedding=ec.embedding,
                        chunk_index=ec.index,
                        token_count=ec.token_count,
                        metadata=ec.metadata,
                    )

                created += 1
                logger.info(
                    f"Image caption chunk created [{location}, img #{img_data.image_index}]: "
                    f"{caption[:80]}..."
                    if len(caption) > 80
                    else f"Image caption chunk created [{location}, img #{img_data.image_index}]: {caption}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to caption image #{img_data.image_index} "
                    f"from {os.path.basename(file_path)}: {e}",
                    exc_info=True,
                )

        return created

    async def _ingest_single_document(
        self,
        session: AsyncSession,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Ingest a single document — text chunks + optional image-caption chunks.

        Args:
            session: Database session
            file_path: Path to document file

        Returns:
            Ingestion result dictionary
        """
        start_time = datetime.now()
        file_ext = os.path.splitext(file_path)[1].lower()

        # --- Standalone image files  ----------------------------------------
        # When image captioning is enabled and the file IS an image, skip text
        # extraction entirely and only run the BLIP captioner.
        standalone_image_exts = {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"
        }
        if file_ext in standalone_image_exts:
            if not settings.image_captioning_enabled:
                logger.info(f"Skipping image file (captioning disabled): {os.path.basename(file_path)}")
                return {"title": os.path.basename(file_path), "chunks_created": 0, "success": False, "error": "Image captioning disabled"}

            title = os.path.splitext(os.path.basename(file_path))[0]
            source = os.path.relpath(file_path, self.documents_folder)

            # Create a parent document record for the image file
            document = await create_document(
                session,
                title=title,
                source=source,
                content=f"[Standalone image file: {os.path.basename(file_path)}]",
                metadata={"file_path": file_path, "file_type": "image", "ingestion_date": datetime.now().isoformat()},
            )

            image_chunks = await self._ingest_image_chunks(
                session, document.id, file_path, base_chunk_index=0
            )
            await session.commit()

            return {
                "title": title,
                "chunks_created": image_chunks,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "success": image_chunks > 0,
                "error": None if image_chunks > 0 else "No captions generated",
            }

        # --- Text / audio / structured documents ----------------------------
        # Read document
        content, docling_doc = self._read_document(file_path)
        title = self._extract_title(content, file_path)
        source = os.path.relpath(file_path, self.documents_folder)

        logger.info(f"Processing: {title}")

        # Text chunking
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=source,
            metadata={"file_path": file_path},
            docling_doc=docling_doc
        )

        if not chunks:
            logger.warning(f"No text chunks created for {title}")

        logger.info(f"Created {len(chunks)} text chunks")

        # Generate embeddings for text chunks
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} text chunks")

        # Save parent document
        document = await create_document(
            session,
            title=title,
            source=source,
            content=content,
            metadata={"file_path": file_path, "ingestion_date": datetime.now().isoformat()}
        )

        # Save text chunks
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

        # Save image-caption chunks (PDF / PPTX / DOCX)
        image_chunks_created = 0
        if file_ext in {".pdf", ".pptx", ".ppt", ".docx", ".doc"}:
            image_chunks_created = await self._ingest_image_chunks(
                session, document.id, file_path, base_chunk_index=len(chunks)
            )

        await session.commit()

        processing_time = (datetime.now() - start_time).total_seconds()
        total_chunks = len(chunks) + image_chunks_created
        logger.info(
            f"Saved document '{title}': {len(chunks)} text chunk(s) + "
            f"{image_chunks_created} image caption chunk(s) in {processing_time:.1f}s"
        )

        # Record metrics
        if METRICS_AVAILABLE:
            metrics.ingestion_documents_total.labels(status="success").inc()
            metrics.ingestion_chunks_created.inc(total_chunks)
            metrics.ingestion_duration.observe(processing_time)

        return {
            "title": title,
            "chunks_created": total_chunks,
            "text_chunks": len(chunks),
            "image_caption_chunks": image_chunks_created,
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
