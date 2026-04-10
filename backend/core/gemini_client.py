"""
Gemini LLM client for chat completions.

Uses the official `google-genai` SDK (v1+) — the successor to the deprecated
`google-generativeai` package.

Provides the same interface as OllamaClient (generate_chat_completion /
generate_chat_completion_stream) so RAGEngine can use Gemini as a drop-in
replacement for Ollama when cloud infra is unavailable.

NOTE: Embeddings are NOT handled here — the system still uses
Ollama/nomic-embed-text for vector search. Only the final answer-generation
step is offloaded to Gemini.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Async wrapper around Google's `google-genai` SDK.

    Only LLM inference methods are implemented; embedding generation
    is left to OllamaClient so the vector-search pipeline is unchanged.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (defaults to settings.gemini_api_key)
            model:   Model name, e.g. 'gemini-1.5-flash' (defaults to settings.gemini_model)
        """
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model

        if not self.api_key:
            raise ValueError(
                "Gemini API key not configured. "
                "Set GEMINI_API_KEY in .env or pass api_key explicitly."
            )

        # Import lazily so google-genai is optional when running Ollama-only mode.
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        self._client = genai.Client(api_key=self.api_key)
        self._types = genai_types

        logger.info(f"Initialized GeminiClient with model={self.model_name}")

    async def generate_chat_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a non-streaming chat completion via Gemini.

        Args:
            prompt:      Full prompt string (system + context + question)
            temperature: Sampling temperature
            max_tokens:  Maximum output tokens

        Returns:
            Generated text string
        """
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            # The SDK's generate_content is synchronous; run in thread pool to
            # avoid blocking the async event loop.
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generate_content failed: {e}")
            raise

    async def generate_chat_completion_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming chat completion via Gemini.

        Yields text chunks as they arrive from the Gemini streaming API.

        Args:
            prompt:      Full prompt string
            temperature: Sampling temperature
            max_tokens:  Maximum output tokens

        Yields:
            Text chunks (strings)
        """
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            # generate_content_stream returns a synchronous iterator.
            # Run in a thread pool and collect chunks to feed the async generator.
            def _stream_sync():
                return list(
                    self._client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                )

            chunks = await asyncio.to_thread(_stream_sync)
            for chunk in chunks:
                try:
                    text = chunk.text
                    if text:
                        yield text
                except Exception:
                    # Some chunks may not carry text (safety / stop chunks)
                    continue

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"❌ Gemini error: {str(e)}"

    async def health_check(self) -> bool:
        """
        Quick connectivity check — tries to list available models.

        Returns:
            True if the API key is valid and reachable, False otherwise.
        """
        try:
            models = await asyncio.to_thread(
                lambda: list(self._client.models.list())
            )
            return len(models) > 0
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False


# ---------------------------------------------------------------------------
# Lazy singleton — instantiated only when Gemini backend is active.
# ---------------------------------------------------------------------------

_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Return the shared GeminiClient instance, creating it on first call."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
