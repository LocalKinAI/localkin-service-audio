"""
Whisper.cpp STT Strategy (via pywhispercpp).
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig, Segment


class WhisperCppStrategy(STTStrategy):
    """
    Speech-to-Text using whisper.cpp (via pywhispercpp).

    Pros:
    - Extremely fast (up to 50x faster than OpenAI Whisper)
    - Very low memory usage
    - Optimized for CPU inference
    - Good for real-time applications

    Cons:
    - Fewer model variants
    - English-focused (though multilingual is supported)
    """

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load whisper.cpp model via pywhispercpp."""
        try:
            from pywhispercpp.model import Model

            model_size = model_config.model_size or "base"

            print(f"Loading whisper.cpp model '{model_size}'...")
            self.model = Model(model_size)

            self.model_config = model_config
            self.device = "cpu"  # whisper.cpp is CPU-optimized
            self._is_loaded = True
            print(f"Whisper.cpp model loaded")
            return True

        except ImportError:
            print("pywhispercpp not installed. Install with: pip install pywhispercpp")
            return False
        except Exception as e:
            print(f"Failed to load whisper.cpp model: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using whisper.cpp."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Load audio as numpy array
            audio_array = self._load_audio(audio)

            # Transcribe
            result = self.model.transcribe(
                audio_array,
                language=language if language else None
            )

            # Extract text from result
            # pywhispercpp returns a list of segments
            segments = []
            full_text = ""

            if isinstance(result, list):
                for seg in result:
                    if hasattr(seg, 'text'):
                        text = seg.text
                        segments.append(Segment(
                            text=text,
                            start=getattr(seg, 't0', 0) / 100.0,  # Convert to seconds
                            end=getattr(seg, 't1', 0) / 100.0,
                        ))
                        full_text += text
                    elif isinstance(seg, dict) and 'text' in seg:
                        text = seg['text']
                        segments.append(Segment(
                            text=text,
                            start=seg.get('t0', 0) / 100.0,
                            end=seg.get('t1', 0) / 100.0,
                        ))
                        full_text += text
            elif isinstance(result, dict) and 'text' in result:
                full_text = result['text']
            elif isinstance(result, str):
                full_text = result

            return TranscriptionResult(
                text=full_text.strip(),
                language=language,
                segments=segments if segments else None,
                model=self.model_config.name if self.model_config else None,
                engine="whisper-cpp",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """whisper.cpp supports same languages as Whisper."""
        from .whisper_strategy import WhisperStrategy
        return WhisperStrategy.get_supported_languages()

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.AVAILABLE_MODELS
