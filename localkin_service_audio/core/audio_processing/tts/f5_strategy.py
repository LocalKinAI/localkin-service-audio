"""
F5-TTS Strategy - Zero-shot voice cloning.

F5-TTS provides high-quality voice cloning with:
- Zero-shot cloning from short reference audio
- Natural prosody
- Multiple language support
"""
from typing import Optional, List, Union
import numpy as np

from .base import TTSStrategy
from ...types import AudioResult, VoiceInfo, ModelConfig


class F5TTSStrategy(TTSStrategy):
    """
    Text-to-Speech using F5-TTS.

    Features:
    - Zero-shot voice cloning
    - High-quality synthesis
    - Natural prosody
    - English and Chinese support

    Requirements:
        pip install f5-tts
    """

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load F5-TTS model."""
        try:
            from f5_tts.api import F5TTS

            self.device = self._detect_device(device)

            print("Loading F5-TTS model...")

            self.model = F5TTS(device=self.device)

            self.model_config = model_config
            self._is_loaded = True

            print(f"F5-TTS loaded on {self.device}")
            return True

        except ImportError:
            print("F5-TTS not installed. Install with: pip install f5-tts")
            return False
        except Exception as e:
            print(f"Failed to load F5-TTS: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize speech.

        Note: F5-TTS requires a reference audio for voice cloning.
        For synthesis without reference, use a default voice sample.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # F5-TTS always needs reference audio
        reference_audio = kwargs.get("reference_audio")
        reference_text = kwargs.get("reference_text", "")

        if not reference_audio:
            raise ValueError(
                "F5-TTS requires reference_audio for synthesis. "
                "Use clone_voice() or provide reference_audio parameter."
            )

        return self.clone_voice(reference_audio, text, reference_text)

    def clone_voice(
        self,
        reference_audio: Union[str, np.ndarray],
        text: str,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> AudioResult:
        """
        Clone voice from reference audio.

        Args:
            reference_audio: Path to reference audio (5-15 seconds recommended)
            text: Text to synthesize with cloned voice
            reference_text: Transcript of reference audio (improves quality)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Load reference audio if numpy array
            if isinstance(reference_audio, np.ndarray):
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, reference_audio, 24000)
                    reference_audio = f.name

            # Generate with voice cloning
            audio, sample_rate = self.model.infer(
                ref_file=reference_audio,
                ref_text=reference_text or "",
                gen_text=text,
            )

            # Convert to numpy if needed
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio = audio.flatten().astype(np.float32)

            return AudioResult(
                audio=audio,
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "f5-tts",
                voice="cloned",
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Voice cloning failed: {e}")

    def list_voices(self) -> List[VoiceInfo]:
        """F5-TTS uses reference audio instead of preset voices."""
        return [
            VoiceInfo(
                id="clone",
                name="Voice Clone",
                language="en",
                description="Clone any voice from reference audio",
                is_cloned=True,
            ),
        ]

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return ["en", "zh"]

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return True

    @classmethod
    def supports_streaming(cls) -> bool:
        return False
