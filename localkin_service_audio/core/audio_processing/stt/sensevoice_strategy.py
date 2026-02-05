"""
SenseVoice STT Strategy - Alibaba FunAudioLLM.

SenseVoice is 15x faster than Whisper with:
- Multilingual support (50+ languages including Chinese, English, Japanese, Korean)
- Emotion detection
- Audio event detection
- Language identification
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig, Segment


class SenseVoiceStrategy(STTStrategy):
    """
    Speech-to-Text using Alibaba's SenseVoice model.

    Features:
    - 15x faster than Whisper
    - Emotion detection (happy, sad, angry, etc.)
    - Audio event detection (laughter, applause, music)
    - Language identification
    - Excellent Chinese and multilingual support

    Requirements:
        pip install funasr modelscope
    """

    AVAILABLE_MODELS = {
        "sensevoice-small": "FunAudioLLM/SenseVoiceSmall",
        "sensevoice-large": "FunAudioLLM/SenseVoiceLarge",
    }

    # Emotion mapping
    EMOTIONS = {
        "HAPPY": "happy",
        "SAD": "sad",
        "ANGRY": "angry",
        "NEUTRAL": "neutral",
        "SURPRISED": "surprised",
        "FEARFUL": "fearful",
    }

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load SenseVoice model."""
        try:
            from funasr import AutoModel

            self.device = self._detect_device(device)
            model_size = model_config.model_size or "small"

            # Get model ID
            model_id = self.AVAILABLE_MODELS.get(
                f"sensevoice-{model_size}",
                model_config.repo_id or "FunAudioLLM/SenseVoiceSmall"
            )

            print(f"Loading SenseVoice model '{model_size}'...")

            # Initialize SenseVoice
            self.model = AutoModel(
                model=model_id,
                trust_remote_code=True,
                device=self.device if self.device != "mps" else "cpu",  # FunASR doesn't support MPS
            )

            self.model_config = model_config
            self._is_loaded = True
            print(f"SenseVoice loaded on {self.device}")
            return True

        except ImportError:
            print("funasr not installed. Install with: pip install funasr modelscope")
            return False
        except Exception as e:
            print(f"Failed to load SenseVoice: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        detect_emotion: bool = True,
        detect_events: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio using SenseVoice.

        Args:
            audio: Audio data or file path
            language: Language code (auto-detect if None)
            detect_emotion: Enable emotion detection
            detect_events: Enable audio event detection
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Prepare audio
            if isinstance(audio, str):
                audio_input = audio
            else:
                # Convert numpy array to file for FunASR
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, 16000)
                    audio_input = f.name

            # Run inference
            result = self.model.generate(
                input=audio_input,
                cache={},
                language=language or "auto",
                use_itn=True,  # Inverse text normalization
            )

            # Parse result
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            # Extract text
            text = result.get("text", "") if isinstance(result, dict) else str(result)

            # Extract emotion if available
            emotion = None
            if detect_emotion and isinstance(result, dict):
                raw_emotion = result.get("emotion", result.get("emo_label"))
                if raw_emotion:
                    emotion = self.EMOTIONS.get(str(raw_emotion).upper(), raw_emotion)

            # Extract audio events if available
            audio_events = None
            if detect_events and isinstance(result, dict):
                events = result.get("event", result.get("audio_events", []))
                if events:
                    audio_events = events if isinstance(events, list) else [events]

            # Extract language
            detected_language = None
            if isinstance(result, dict):
                detected_language = result.get("language", result.get("lang"))
                if detected_language:
                    detected_language = detected_language.lower()[:2]

            return TranscriptionResult(
                text=text.strip(),
                language=detected_language or language,
                emotion=emotion,
                audio_events=audio_events,
                model=self.model_config.name if self.model_config else "sensevoice",
                engine="sensevoice",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """SenseVoice supports 50+ languages."""
        return [
            "zh", "en", "ja", "ko", "yue",  # Primary
            "de", "es", "fr", "it", "pt", "ru",  # European
            "ar", "hi", "th", "vi", "id", "ms",  # Asian
            # ... and many more
        ]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.AVAILABLE_MODELS.keys())

    @classmethod
    def supports_emotion_detection(cls) -> bool:
        return True

    @classmethod
    def supports_event_detection(cls) -> bool:
        return True
