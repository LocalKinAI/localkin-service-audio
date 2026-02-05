"""
Kokoro TTS Strategy - High-quality neural TTS.
"""
from typing import Optional, List
import numpy as np

from .base import TTSStrategy
from ...types import AudioResult, VoiceInfo, ModelConfig


class KokoroStrategy(TTSStrategy):
    """
    Text-to-Speech using Kokoro TTS.

    Pros:
    - High-quality neural TTS
    - Fast inference
    - Multiple voices
    - Good prosody

    Cons:
    - Requires model download
    - Higher resource usage than native
    """

    AVAILABLE_MODELS = ["kokoro-82m", "kokoro-128m"]

    # Kokoro voice presets
    VOICES = {
        "af": "American Female",
        "af_bella": "Bella (American Female)",
        "af_sarah": "Sarah (American Female)",
        "am_adam": "Adam (American Male)",
        "am_michael": "Michael (American Male)",
        "bf_emma": "Emma (British Female)",
        "bf_isabella": "Isabella (British Female)",
        "bm_george": "George (British Male)",
        "bm_lewis": "Lewis (British Male)",
    }

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load Kokoro TTS model."""
        try:
            import kokoro

            self.device = self._detect_device(device)
            model_name = model_config.model_size or "kokoro-82m"

            print(f"Loading Kokoro TTS model '{model_name}'...")

            # Initialize Kokoro
            self.model = kokoro.KPipeline(lang_code='a')  # 'a' for American English

            self.model_config = model_config
            self._is_loaded = True
            self._current_voice = "af"  # Default voice

            print(f"Kokoro TTS loaded")
            return True

        except ImportError:
            print("Kokoro not installed. Install with: pip install kokoro")
            return False
        except Exception as e:
            print(f"Failed to load Kokoro model: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """Synthesize speech using Kokoro."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            voice_id = voice or self._current_voice or "af"

            # Generate audio
            generator = self.model(
                text,
                voice=voice_id,
                speed=speed,
            )

            # Collect all audio segments
            audio_segments = []
            sample_rate = 24000  # Kokoro default

            for _, _, audio in generator:
                if audio is not None:
                    audio_segments.append(audio)

            # Concatenate segments
            if audio_segments:
                audio = np.concatenate(audio_segments)
            else:
                audio = np.array([], dtype=np.float32)

            return AudioResult(
                audio=audio,
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "kokoro",
                voice=voice_id,
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def list_voices(self) -> List[VoiceInfo]:
        """List available Kokoro voices."""
        voices = []
        for voice_id, name in self.VOICES.items():
            lang = "en"
            gender = "female" if voice_id.startswith(("af", "bf")) else "male"

            voices.append(VoiceInfo(
                id=voice_id,
                name=name,
                language=lang,
                gender=gender,
                sample_rate=24000,
            ))
        return voices

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return ["en"]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.AVAILABLE_MODELS

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return False

    @classmethod
    def supports_streaming(cls) -> bool:
        return True  # Kokoro yields audio segments
