"""
MusicGen strategy - Music generation using Hugging Face's MusicGen model.

MusicGen is a simple and controllable music generation model capable of
generating high-quality music samples conditioned on text descriptions or
melody in the style of a reference track.

Model: facebook/musicgen-{small|medium|large}
Source: https://huggingface.co/facebook/musicgen-small

Note: Uses direct model.generate() instead of pipeline for better quality control.
The key is do_sample=True which enables proper sampling for richer audio output.
"""
import numpy as np
import logging
from typing import Optional
import os
import tempfile

from .base import MusicEngine
from ..core.types import AudioResult, ModelConfig

logger = logging.getLogger(__name__)


class MusicGenStrategy(MusicEngine):
    """
    Music generation using Meta's MusicGen model.

    Supports model sizes: small (1.5GB), medium (3GB), large (15GB)
    
    Uses direct model.generate() for better quality control.
    Key: do_sample=True enables proper sampling for richer audio.
    """

    def __init__(self):
        super().__init__()
        self.model_size: Optional[str] = None
        self.processor = None
        # Note: self.model is inherited from base class

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """
        Load MusicGen model using direct model loading (not pipeline).

        Args:
            model_config: Configuration with model_name like "musicgen-small"
            device: Device to use ("auto", "cpu", "cuda", "mps")

        Returns:
            True if loaded successfully
        """
        try:
            import torch
            from transformers import MusicgenForConditionalGeneration, AutoProcessor

            self.device = self._detect_device(device)
            self.model_config = model_config

            # Extract model size from config (e.g., "musicgen-small" -> "small")
            model_name = model_config.name
            if ":" in model_name:
                # Handle "musicgen:small" format
                self.model_size = model_name.split(":")[-1].lower()
            elif "-" in model_name:
                # Handle "musicgen-small" format
                self.model_size = model_name.split("-")[-1].lower()
            else:
                # Default to small
                self.model_size = "small"

            if self.model_size not in ["small", "medium", "large"]:
                logger.error(f"Unsupported model size: {self.model_size}")
                return False

            # Build HuggingFace model ID
            huggingface_model_id = f"facebook/musicgen-{self.model_size}"

            logger.info(f"Loading MusicGen model: {huggingface_model_id}")
            logger.info(f"Device: {self.device}")

            # Load model and processor directly (not via pipeline)
            # This gives us full control over generation parameters
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                huggingface_model_id
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(huggingface_model_id)

            self._is_loaded = True
            logger.info(f"MusicGen {self.model_size} loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load MusicGen model: {e}", exc_info=True)
            self._is_loaded = False
            return False

    def generate(
        self,
        prompt: str,
        duration: int = 10,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        **kwargs
    ) -> AudioResult:
        """
        Generate music from text prompt using direct model.generate() call.

        Args:
            prompt: Text description of music
            duration: Duration in seconds (5-30 seconds)
            temperature: Sampling temperature (1.0 default)
            top_k: Top-k sampling parameter (250 default)
            top_p: Top-p sampling parameter (0.0 default, disabled)
            **kwargs: Additional arguments

        Returns:
            AudioResult with generated audio
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if duration not in [5, 10, 15, 20, 30]:
            # Find closest supported duration
            supported = [5, 10, 15, 20, 30]
            duration = min(supported, key=lambda x: abs(x - duration))
            logger.warning(f"Duration {duration} not supported, using {duration}s")

        try:
            import torch

            logger.info(f"Generating music: '{prompt}' ({duration}s)")
            logger.info("Using direct model.generate() with do_sample=True (proper sampling)")

            # Process text input using the processor
            # This creates the proper input format for the model
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            logger.info(f"Inputs prepared. Max new tokens: {duration * 50}")

            # Generate audio using the model directly
            # KEY: do_sample=True is ESSENTIAL for quality output!
            # Without it, greedy decoding produces monotonous, low-quality audio
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,           # ⭐ CRITICAL: Enable proper sampling
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p if top_p > 0 else None,
                    max_new_tokens=int(duration * 50),  # ~50 tokens per second
                    guidance_scale=3.0,       # For better quality
                )

            # Extract audio from generation output
            # audio_values shape: (batch_size, num_channels, num_samples)
            audio_np = audio_values[0][0].cpu().numpy()  # Take first batch, first channel
            
            # Ensure float32 format
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)

            # MusicGen outputs at 16kHz by default
            sampling_rate = 16000

            actual_duration = len(audio_np) / sampling_rate
            logger.info(f"Generated {actual_duration:.2f}s of audio at {sampling_rate}Hz")

            return AudioResult(
                audio=audio_np,
                sample_rate=sampling_rate,
                model=f"musicgen-{self.model_size}",
                duration=actual_duration,
            )

        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)
            raise

    def unload(self) -> None:
        """Release model resources."""
        if self.model is not None:
            # Try to free GPU memory
            try:
                del self.model
                import gc
                gc.collect()

                # If using CUDA, clear cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

        self.model = None
        self.processor = None
        self._is_loaded = False
        logger.info("MusicGen model unloaded")

    def get_info(self):
        """Return model information."""
        info = super().get_info()
        info["model_size"] = self.model_size
        return info

    @classmethod
    def get_model_sizes(cls) -> list:
        """Return list of supported model sizes."""
        return ["small", "medium", "large"]

    @classmethod
    def get_supported_durations(cls) -> list:
        """Return list of supported durations in seconds."""
        return [5, 10, 15, 20, 30]

    @classmethod
    def get_model_requirements(cls) -> dict:
        """Return memory requirements for each model size."""
        return {
            "small": {"vram_gb": 2, "ram_gb": 4, "disk_gb": 2},
            "medium": {"vram_gb": 4, "ram_gb": 8, "disk_gb": 4},
            "large": {"vram_gb": 16, "ram_gb": 16, "disk_gb": 15},
        }
