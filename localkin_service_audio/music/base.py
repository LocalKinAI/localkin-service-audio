"""
Base class for Music Generation strategies.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time

from ..core.types import AudioResult, ModelConfig


class MusicEngine(ABC):
    """
    Abstract base class for Music Generation engines.

    Each music generation engine (MusicGen, Jukebox, etc.)
    implements this interface.
    """

    def __init__(self):
        self.model = None
        self.model_config: Optional[ModelConfig] = None
        self.device: str = "cpu"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded and self.model is not None

    @abstractmethod
    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """
        Load the model into memory.

        Args:
            model_config: Configuration for the model
            device: Device to load model on ("auto", "cpu", "cuda", "mps")

        Returns:
            True if loaded successfully
        """
        pass

    @abstractmethod
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
        Generate music from text prompt.

        Args:
            prompt: Text description of the music to generate
            duration: Duration of generated music in seconds (default: 10)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            **kwargs: Additional engine-specific arguments

        Returns:
            AudioResult with generated audio
        """
        pass

    def generate_to_file(
        self,
        prompt: str,
        output_path: str,
        duration: int = 10,
        **kwargs
    ) -> str:
        """
        Generate music and save to file.

        Args:
            prompt: Text description of the music
            output_path: Path to save audio file
            duration: Duration in seconds
            **kwargs: Additional arguments

        Returns:
            Path to saved file
        """
        result = self.generate(prompt, duration=duration, **kwargs)
        return result.save(output_path)

    def unload(self) -> None:
        """Release model resources."""
        self.model = None
        self._is_loaded = False

    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "engine": self.__class__.__name__,
            "model_config": self.model_config,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }

    def _detect_device(self, requested: str = "auto") -> str:
        """
        Detect the best available device.

        Args:
            requested: Requested device ("auto", "cpu", "cuda", "mps")

        Returns:
            Device string
        """
        if requested != "auto":
            return requested

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @classmethod
    def supports_streaming(cls) -> bool:
        """Return whether this engine supports streaming generation."""
        return False

    @classmethod
    def get_supported_durations(cls) -> list:
        """Return list of supported durations in seconds."""
        return [5, 10, 15, 20, 30]  # Override in subclasses

    @classmethod
    def get_model_sizes(cls) -> list:
        """Return list of supported model sizes."""
        return ["small", "medium", "large"]  # Override in subclasses
