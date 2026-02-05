"""
Base class for Speech-to-Text strategies.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
import numpy as np
import time

from ...types import TranscriptionResult, ModelConfig, Segment


class STTStrategy(ABC):
    """
    Abstract base class for Speech-to-Text strategies.

    Each STT engine (Whisper, faster-whisper, whisper.cpp, SenseVoice, etc.)
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
    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (16kHz mono) or path to audio file
            language: Optional language code (e.g., "en", "zh")
            **kwargs: Additional engine-specific arguments

        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        pass

    def transcribe_file(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            TranscriptionResult
        """
        return self.transcribe(audio_path, **kwargs)

    def transcribe_stream(
        self,
        audio_stream,
        chunk_size: int = 16000,
        **kwargs
    ):
        """
        Transcribe streaming audio (generator).

        Args:
            audio_stream: Iterator/generator yielding audio chunks
            chunk_size: Size of each chunk in samples
            **kwargs: Additional arguments

        Yields:
            TranscriptionResult for each chunk (partial results)
        """
        # Default implementation: buffer and transcribe
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) >= chunk_size:
                result = self.transcribe(buffer, **kwargs)
                yield result
                buffer = np.array([], dtype=np.float32)

        # Final chunk
        if len(buffer) > 0:
            yield self.transcribe(buffer, **kwargs)

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

    def _load_audio(self, audio: Union[np.ndarray, str]) -> np.ndarray:
        """
        Load audio from file or return numpy array.

        Args:
            audio: Audio data or path to file

        Returns:
            Audio as numpy array (16kHz, mono, float32)
        """
        if isinstance(audio, str):
            import librosa
            audio_array, _ = librosa.load(audio, sr=16000, mono=True)
            return audio_array.astype(np.float32)

        return audio.astype(np.float32)

    def _time_transcription(self, func):
        """Decorator to time transcription."""
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
    def get_available_models(cls) -> List[str]:
        """Return list of available model sizes/variants."""
        return []  # Override in subclasses
