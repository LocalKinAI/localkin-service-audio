"""
OpenAI Whisper STT Strategy.
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig, Segment


class WhisperStrategy(STTStrategy):
    """
    Speech-to-Text using OpenAI's original Whisper implementation.

    Pros:
    - Most accurate
    - Best multilingual support
    - Reference implementation

    Cons:
    - Slower than alternatives
    - Higher memory usage
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load OpenAI Whisper model."""
        try:
            import whisper

            self.device = self._detect_device(device)
            model_size = model_config.model_size or "base"

            print(f"Loading OpenAI Whisper model '{model_size}'...")
            self.model = whisper.load_model(model_size, device=self.device)

            self.model_config = model_config
            self._is_loaded = True
            print(f"Whisper model loaded on {self.device}")
            return True

        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Whisper accepts file paths directly
            if isinstance(audio, str):
                audio_input = audio
            else:
                audio_input = self._load_audio(audio)

            # Build transcription options
            options = {
                "fp16": self.device == "cuda",  # Use FP16 on CUDA
                "language": language,
            }
            options.update(kwargs)

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            result = self.model.transcribe(audio_input, **options)

            # Build segments
            segments = []
            for seg in result.get("segments", []):
                segments.append(Segment(
                    text=seg["text"],
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("avg_logprob"),
                ))

            return TranscriptionResult(
                text=result["text"].strip(),
                language=result.get("language"),
                segments=segments if segments else None,
                model=self.model_config.name if self.model_config else None,
                engine="whisper",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Whisper supports 99 languages."""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
        ]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.AVAILABLE_MODELS
