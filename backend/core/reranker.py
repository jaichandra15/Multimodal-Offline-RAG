"""
Cross-encoder reranker for improving retrieval quality.

Uses a pretrained cross-encoder model to rescore query-document pairs
and provide more accurate relevance ranking than initial retrieval.
"""

import logging
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from backend.config import settings
from backend.database.operations import SearchResult

try:
    from backend.core.observability import metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for the reranker."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    max_length: int = 512
    

class Reranker:
    """
    Cross-encoder reranker for query-document pairs.
    
    Provides more accurate relevance scoring than bi-encoder embeddings
    by encoding query and document together.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize reranker.
        
        Args:
            config: Reranker configuration
        """
        self.config = config or RerankerConfig()
        self.model: Optional[CrossEncoder] = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy load the model on first use."""
        if self._initialized:
            return
            
        logger.info(f"Loading reranker model: {self.config.model_name}")
        logger.info(f"Using device: {self.config.device}")
        
        try:
            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device
            )
            self._initialized = True
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}", exc_info=True)
            raise
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: User query
            results: Initial search results from hybrid/vector search
            top_k: Number of top results to return (defaults to len(results))
            
        Returns:
            Reranked list of SearchResult instances with updated similarity scores
        """
        if not results:
            return results
        
        # Ensure model is loaded
        self._ensure_initialized()
        
        if not self.model:
            logger.warning("Reranker model not available, returning original results")
            return results
        
        start_time = time.time()
        
        try:
            # Prepare query-document pairs
            pairs = [(query, result.content) for result in results]
            
            # Get reranker scores
            logger.info(f"Reranking {len(pairs)} results...")
            scores = self.model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )
            
            # Update results with new scores
            reranked_results = []
            for result, score in zip(results, scores):
                # Create new result with reranker score
                result.similarity = float(score)
                reranked_results.append(result)
            
            # Sort by new scores (descending)
            reranked_results.sort(key=lambda x: x.similarity, reverse=True)
            
            # Take top K if specified
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
            
            rerank_duration = time.time() - start_time
            
            # Record metrics
            if METRICS_AVAILABLE:
                metrics.reranker_latency.observe(rerank_duration)
                
                # Calculate ranking change (position delta for top result)
                if len(results) > 1:
                    original_top_id = results[0].chunk_id
                    new_top_id = reranked_results[0].chunk_id
                    if original_top_id != new_top_id:
                        original_position = next(
                            (i for i, r in enumerate(results) if r.chunk_id == new_top_id),
                            len(results)
                        )
                        metrics.reranker_rank_change.observe(original_position)
            
            logger.info(
                f"Reranking completed in {rerank_duration:.3f}s, "
                f"top score: {reranked_results[0].similarity:.4f}"
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Fallback to original results
            return results
    
    def score_pair(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.
        
        Args:
            query: Query text
            text: Document text
            
        Returns:
            Relevance score
        """
        self._ensure_initialized()
        
        if not self.model:
            return 0.0
        
        try:
            score = self.model.predict([(query, text)])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0


# Global reranker instance (lazy loaded)
_reranker_instance: Optional[Reranker] = None

def get_reranker() -> Reranker:
    """Get or create global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        config = RerankerConfig(
            model_name=settings.reranker_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=settings.reranker_batch_size,
        )
        _reranker_instance = Reranker(config)
    return _reranker_instance
