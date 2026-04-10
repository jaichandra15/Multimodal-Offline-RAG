"""
BLIP-based image captioner for multimodal RAG ingestion.

Uses Salesforce BLIP (Bootstrapping Language-Image Pre-training) to generate
natural-language captions for images extracted from PDFs, PPTXs, or uploaded
as standalone files.

The model is **lazy-loaded** on first use so that the import of this module
has zero cost when image captioning is disabled.

Supported model variants (lightest → heaviest):
  - Salesforce/blip-image-captioning-base   (~940 MB download, ~1 GB RAM)  ← default
  - Salesforce/blip-image-captioning-large  (~1.9 GB download, ~2.5 GB RAM)
  - Salesforce/blip2-opt-2.7b              (~6 GB download, ~10 GB RAM)

Hardware acceleration:
  - Apple M1/M2/M3  → MPS (Metal Performance Shaders)
  - NVIDIA GPU      → CUDA
  - Fallback        → CPU
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class BLIPCaptioner:
    """
    Wraps a Salesforce BLIP model to caption PIL images.

    Parameters
    ----------
    model_name
        HuggingFace model identifier.  Defaults to the base BLIP captioning
        model which gives a good quality / speed balance on CPU and MPS.
    max_new_tokens
        Maximum number of tokens in the generated caption sentence.
    conditional_prompt
        Optional text prompt to condition the caption.  E.g.
        ``"a diagram showing"`` can help steer the model for technical docs.
        Leave as ``None`` for pure unconditional captioning.
    """

    DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 80,
        conditional_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.conditional_prompt = conditional_prompt

        # Resolved lazily on first caption() call
        self._processor = None
        self._model = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Download and initialise model + processor if not already done."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "transformers and/or torch are not installed. "
                "Install with: pip install transformers torch Pillow"
            )

        # Pick best available device
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(
            f"BLIPCaptioner: loading '{self.model_name}' onto {self._device} …"
            "  (first run downloads ~1 GB — subsequent runs use cache)"
        )

        t0 = time.time()
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self._device)
        self._model.eval()

        logger.info(
            f"BLIPCaptioner: model loaded in {time.time() - t0:.1f}s "
            f"on {self._device}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def caption(self, image) -> str:
        """
        Generate a natural-language caption for *image*.

        Parameters
        ----------
        image
            A ``PIL.Image.Image`` instance.

        Returns
        -------
        str
            The generated caption, e.g.
            ``"a bar chart showing quarterly revenue by region"``.
            Returns an empty string on failure so the caller can decide
            whether to skip the chunk.
        """
        self._load()

        try:
            import torch

            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self._processor(
                images=image,
                text=self.conditional_prompt,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            caption: str = self._processor.decode(
                output_ids[0], skip_special_tokens=True
            ).strip()

            logger.debug(f"BLIPCaptioner: '{caption}'")
            return caption

        except Exception as e:
            logger.error(f"BLIPCaptioner: captioning failed — {e}", exc_info=True)
            return ""

    def caption_batch(self, images: list) -> list[str]:
        """
        Caption multiple images in one call (more efficient on GPU).

        Falls back to sequential captioning on CPU/MPS where batching
        gives minimal throughput gain.
        """
        self._load()

        try:
            import torch

            rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

            inputs = self._processor(
                images=rgb_images,
                text=[self.conditional_prompt] * len(rgb_images) if self.conditional_prompt else None,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            captions = [
                self._processor.decode(ids, skip_special_tokens=True).strip()
                for ids in output_ids
            ]
            return captions

        except Exception as e:
            logger.warning(
                f"BLIPCaptioner: batch captioning failed ({e}), falling back to sequential"
            )
            return [self.caption(img) for img in images]

    def unload(self) -> None:
        """Release GPU/CPU memory by discarding the model."""
        if self._model is not None:
            import torch

            del self._model
            del self._processor
            self._model = None
            self._processor = None

            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps":
                torch.mps.empty_cache()

            logger.info("BLIPCaptioner: model unloaded from memory")


# ---------------------------------------------------------------------------
# Module-level singleton — created lazily, shared across ingestion runs
# ---------------------------------------------------------------------------

_captioner_instance: Optional[BLIPCaptioner] = None


def get_captioner(model_name: Optional[str] = None) -> BLIPCaptioner:
    """
    Return the shared ``BLIPCaptioner`` singleton.

    The model is resolved from (in order of priority):
    1. *model_name* argument
    2. ``BLIP_MODEL`` environment variable
    3. ``BLIPCaptioner.DEFAULT_MODEL``
    """
    global _captioner_instance

    resolved_model = (
        model_name
        or os.getenv("BLIP_MODEL")
        or BLIPCaptioner.DEFAULT_MODEL
    )

    if _captioner_instance is None or _captioner_instance.model_name != resolved_model:
        if _captioner_instance is not None:
            _captioner_instance.unload()
        _captioner_instance = BLIPCaptioner(model_name=resolved_model)

    return _captioner_instance
