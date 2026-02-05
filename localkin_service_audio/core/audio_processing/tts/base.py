"""
Base class for Text-to-Speech strategies.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import numpy as np
import time

from ...types import AudioResult, VoiceInfo, ModelConfig


class TTSStrategy(ABC):
    """
    Abstract base class for Text-to-Speech strategies.

    Each TTS engine (pyttsx3, Kokoro, CosyVoice, ChatTTS, etc.)
    implements this interface.
    """

    def __init__(self):
        self.model = None
        self.model_config: Optional[ModelConfig] = None
        self.device: str = "cpu"
        self._is_loaded: bool = False
        self._current_voice: Optional[str] = None

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
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID to use (None for default)
            speed: Speech speed multiplier (1.0 = normal)
            **kwargs: Additional engine-specific arguments

        Returns:
            AudioResult with synthesized audio
        """
        pass

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Synthesize speech and save to file.

        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            voice: Voice ID to use
            **kwargs: Additional arguments

        Returns:
            Path to saved file
        """
        result = self.synthesize(text, voice=voice, **kwargs)
        return result.save(output_path)

    def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        chunk_size: int = 4096,
        **kwargs
    ):
        """
        Synthesize speech as a stream (generator).

        Args:
            text: Text to synthesize
            voice: Voice ID to use
            chunk_size: Size of each audio chunk
            **kwargs: Additional arguments

        Yields:
            Audio chunks as numpy arrays
        """
        # Default implementation: synthesize all and chunk
        result = self.synthesize(text, voice=voice, **kwargs)
        audio = result.audio

        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def clone_voice(
        self,
        reference_audio: Union[str, np.ndarray],
        text: str,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> AudioResult:
        """
        Clone a voice from reference audio and synthesize text.

        Args:
            reference_audio: Path to reference audio or audio array
            text: Text to synthesize with cloned voice
            reference_text: Transcript of reference audio (for some models)
            **kwargs: Additional arguments

        Returns:
            AudioResult with synthesized audio

        Raises:
            NotImplementedError: If voice cloning is not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support voice cloning"
        )

    def list_voices(self) -> List[VoiceInfo]:
        """
        List available voices.

        Returns:
            List of VoiceInfo objects
        """
        return []

    def set_voice(self, voice: str) -> bool:
        """
        Set the current voice.

        Args:
            voice: Voice ID to set

        Returns:
            True if voice was set successfully
        """
        voices = self.list_voices()
        voice_ids = [v.id for v in voices]

        if voice in voice_ids:
            self._current_voice = voice
            return True
        return False

    def unload(self) -> None:
        """Release model resources."""
        self.model = None
        self._is_loaded = False
        self._current_voice = None

    def get_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "engine": self.__class__.__name__,
            "model_config": self.model_config,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "current_voice": self._current_voice,
            "voices": [v.id for v in self.list_voices()],
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

    def _time_synthesis(self, func):
        """Decorator to time synthesis."""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            result.processing_time = time.time() - start
            return result
        return wrapper

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Return list of supported languages."""
        return ["en"]  # Override in subclasses

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        """Return whether this engine supports voice cloning."""
        return False

    @classmethod
    def supports_streaming(cls) -> bool:
        """Return whether this engine supports streaming synthesis."""
        return False

    @classmethod
    def supports_emotion(cls) -> bool:
        """Return whether this engine supports emotion control."""
        return False
