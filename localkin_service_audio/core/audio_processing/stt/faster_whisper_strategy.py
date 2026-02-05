"""
Faster Whisper STT Strategy (CTranslate2 implementation).
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig, Segment


class FasterWhisperStrategy(STTStrategy):
    """
    Speech-to-Text using faster-whisper (CTranslate2 implementation).

    Pros:
    - Up to 4x faster than OpenAI Whisper
    - Lower memory usage with int8 quantization
    - VAD (Voice Activity Detection) support
    - Batched inference for long audio

    Cons:
    - No MPS (Apple Silicon) support - CPU only on Mac
    - Slightly lower accuracy than original
    """

    AVAILABLE_MODELS = [
        "tiny", "base", "small", "medium",
        "large-v2", "large-v3", "turbo", "distil-large-v3"
    ]

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            # faster-whisper only supports CPU and CUDA (not MPS)
            requested_device = self._detect_device(device)
            if requested_device == "mps":
                print("Warning: faster-whisper doesn't support MPS. Using CPU.")
                self.device = "cpu"
            elif requested_device == "cuda":
                self.device = "cuda"
            else:
                self.device = "cpu"

            model_size = model_config.model_size or "base"

            # Determine compute type based on device
            if self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"

            print(f"Loading faster-whisper model '{model_size}' on {self.device}...")
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type
            )

            self.model_config = model_config
            self._is_loaded = True
            print(f"Faster-whisper model loaded ({compute_type})")
            return True

        except ImportError:
            print("faster-whisper not installed. Install with: pip install faster-whisper")
            return False
        except Exception as e:
            print(f"Failed to load faster-whisper model: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        enable_vad: bool = True,
        beam_size: int = 5,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using faster-whisper."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Load audio if path provided
            if isinstance(audio, str):
                audio_path = audio
            else:
                # Save to temp file for faster-whisper
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, 16000)
                    audio_path = f.name

            # Check audio duration for batched inference
            duration = self._get_audio_duration(audio_path)

            # VAD parameters
            vad_params = None
            if enable_vad:
                vad_params = {"min_silence_duration_ms": 500}

            # Use batched inference for long audio (>5 minutes)
            if duration > 300:
                from faster_whisper import BatchedInferencePipeline
                batched_model = BatchedInferencePipeline(model=self.model)
                segments_iter, info = batched_model.transcribe(
                    audio_path,
                    batch_size=4,
                    beam_size=beam_size,
                    language=language,
                    vad_filter=enable_vad,
                    vad_parameters=vad_params,
                )
            else:
                segments_iter, info = self.model.transcribe(
                    audio_path,
                    beam_size=beam_size,
                    language=language,
                    vad_filter=enable_vad,
                    vad_parameters=vad_params,
                )

            # Collect segments
            segments = []
            full_text = ""
            for seg in segments_iter:
                segments.append(Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                ))
                full_text += seg.text

            return TranscriptionResult(
                text=full_text.strip(),
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                segments=segments if segments else None,
                model=self.model_config.name if self.model_config else None,
                engine="faster-whisper",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            import contextlib
            with contextlib.closing(wave.open(audio_path, 'r')) as f:
                return f.getnframes() / float(f.getframerate())
        except:
            return 0

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Same as Whisper - 99 languages."""
        from .whisper_strategy import WhisperStrategy
        return WhisperStrategy.get_supported_languages()

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.AVAILABLE_MODELS
