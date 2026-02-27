"""
Ollama HTTP client for embeddings and chat completions.
Handles all communication with the Ollama API.
"""

import httpx
import json
import logging
from typing import List, AsyncGenerator, Optional, Dict, Any

from backend.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL (defaults to settings)
            llm_model: LLM model name (defaults to settings)
            embedding_model: Embedding model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
        """
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self.llm_model = llm_model or settings.ollama_llm_model
        self.embedding_model = embedding_model or settings.ollama_embedding_model
        self.timeout = timeout or settings.ollama_timeout
        
        logger.info(
            f"Initialized OllamaClient: {self.base_url}, "
            f"LLM={self.llm_model}, Embeddings={self.embedding_model}"
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama embedding API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    async def generate_chat_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate chat completion (non-streaming).
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
                
        except httpx.ReadTimeout:
            logger.error("Ollama chat completion timed out")
            return "⏱️ The model took too long to respond. Please try a shorter question."
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama chat API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {e}")
            raise
    
    async def generate_chat_completion_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """
        Generate chat completion with streaming.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks as they are generated
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                ) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.ReadTimeout:
            logger.error("Ollama streaming timed out")
            yield "⏱️ The model took too long to respond."
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"❌ Error: {str(e)}"
    
    async def health_check(self) -> bool:
        """
        Check if Ollama is available and responding.
        
        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.
        
        Returns:
            List of model information dictionaries
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Global Ollama client instance
ollama_client = OllamaClient()
