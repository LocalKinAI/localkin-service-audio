"""
Native OS TTS Strategy (via pyttsx3).
"""
from typing import Optional, List
import numpy as np
import tempfile
import os

from .base import TTSStrategy
from ...types import AudioResult, VoiceInfo, ModelConfig


class NativeStrategy(TTSStrategy):
    """
    Text-to-Speech using native OS engines via pyttsx3.

    Pros:
    - No model download required
    - Works offline
    - Very fast
    - Low resource usage

    Cons:
    - Lower quality than neural TTS
    - Limited voice options
    - Robotic sounding
    """

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3

            self.model = pyttsx3.init()
            self.model_config = model_config
            self.device = "cpu"
            self._is_loaded = True

            # Get default voice
            voices = self.model.getProperty('voices')
            if voices:
                self._current_voice = voices[0].id

            print("Native TTS engine initialized")
            return True

        except Exception as e:
            print(f"Failed to initialize native TTS: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """Synthesize speech using native engine."""
        if not self.is_loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Set voice if specified
            if voice:
                self.model.setProperty('voice', voice)
            elif self._current_voice:
                self.model.setProperty('voice', self._current_voice)

            # Set rate (default is ~200 wpm)
            default_rate = self.model.getProperty('rate')
            self.model.setProperty('rate', int(default_rate * speed))

            # Save to temp file and read back
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            self.model.save_to_file(text, temp_path)
            self.model.runAndWait()

            # Read audio file
            import soundfile as sf
            audio, sample_rate = sf.read(temp_path)

            # Clean up temp file
            os.unlink(temp_path)

            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            return AudioResult(
                audio=audio,
                sample_rate=sample_rate,
                model="native",
                voice=voice or self._current_voice,
                duration=len(audio) / sample_rate,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def list_voices(self) -> List[VoiceInfo]:
        """List available system voices."""
        if not self.is_loaded:
            return []

        voices = []
        try:
            system_voices = self.model.getProperty('voices')
            for v in system_voices:
                # Parse language from voice info
                lang = "en"
                if hasattr(v, 'languages') and v.languages:
                    lang = v.languages[0][:2] if v.languages[0] else "en"

                voices.append(VoiceInfo(
                    id=v.id,
                    name=v.name,
                    language=lang,
                    gender=getattr(v, 'gender', None),
                ))
        except:
            pass

        return voices

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Depends on installed system voices."""
        return ["en", "zh", "ja", "de", "fr", "es"]  # Common ones

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return False

    @classmethod
    def supports_streaming(cls) -> bool:
        return False
