"""
CosyVoice TTS Strategy - Alibaba's SOTA Chinese TTS.

CosyVoice 2.0 achieves state-of-the-art performance with:
- High-fidelity voice cloning
- Cross-lingual synthesis
- Streaming support
- Multiple built-in voices
"""
from typing import Optional, List, Union
import numpy as np

from .base import TTSStrategy
from ...types import AudioResult, VoiceInfo, ModelConfig


class CosyVoiceStrategy(TTSStrategy):
    """
    Text-to-Speech using Alibaba's CosyVoice 2.0.

    Features:
    - State-of-the-art Chinese TTS quality
    - Zero-shot voice cloning
    - Cross-lingual synthesis (speak Chinese with English voice)
    - Streaming support
    - Multiple built-in voices
    - Emotion control via instruct model

    Requirements:
        pip install cosyvoice
        # or install from source: https://github.com/FunAudioLLM/CosyVoice
    """

    AVAILABLE_MODELS = {
        "cosyvoice-300m": "iic/CosyVoice-300M",
        "cosyvoice-300m-sft": "iic/CosyVoice-300M-SFT",
        "cosyvoice-300m-instruct": "iic/CosyVoice-300M-Instruct",
        "cosyvoice-ttsfrd": "iic/CosyVoice-ttsfrd",
    }

    BUILTIN_VOICES = {
        "中文女": {"language": "zh", "gender": "female", "description": "Chinese Female"},
        "中文男": {"language": "zh", "gender": "male", "description": "Chinese Male"},
        "英文女": {"language": "en", "gender": "female", "description": "English Female"},
        "英文男": {"language": "en", "gender": "male", "description": "English Male"},
        "日语男": {"language": "ja", "gender": "male", "description": "Japanese Male"},
        "粤语女": {"language": "yue", "gender": "female", "description": "Cantonese Female"},
        "韩语女": {"language": "ko", "gender": "female", "description": "Korean Female"},
    }

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load CosyVoice model."""
        try:
            from cosyvoice import CosyVoice

            self.device = self._detect_device(device)
            model_size = model_config.model_size or "300m-sft"

            # Get model ID
            model_key = f"cosyvoice-{model_size}"
            model_id = self.AVAILABLE_MODELS.get(
                model_key,
                model_config.repo_id or self.AVAILABLE_MODELS["cosyvoice-300m-sft"]
            )

            print(f"Loading CosyVoice model '{model_size}'...")

            self.model = CosyVoice(model_id)
            self.model_config = model_config
            self._is_loaded = True
            self._current_voice = "中文女"

            print(f"CosyVoice loaded")
            return True

        except ImportError:
            print("CosyVoice not installed. Install from: https://github.com/FunAudioLLM/CosyVoice")
            return False
        except Exception as e:
            print(f"Failed to load CosyVoice: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize speech using CosyVoice.

        Args:
            text: Text to synthesize
            voice: Voice ID (built-in voice name like "中文女")
            speed: Speech speed multiplier
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            voice_id = voice or self._current_voice or "中文女"

            # Use SFT inference for built-in voices
            generator = self.model.inference_sft(text, voice_id)

            # Collect audio
            audio_segments = []
            sample_rate = 22050  # CosyVoice default

            for segment in generator:
                if isinstance(segment, dict) and 'tts_speech' in segment:
                    audio = segment['tts_speech'].numpy()
                    audio_segments.append(audio)
                elif isinstance(segment, np.ndarray):
                    audio_segments.append(segment)

            if audio_segments:
                audio = np.concatenate(audio_segments)
            else:
                audio = np.array([], dtype=np.float32)

            # Apply speed adjustment if needed
            if speed != 1.0:
                audio = self._adjust_speed(audio, sample_rate, speed)

            return AudioResult(
                audio=audio.flatten(),
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "cosyvoice",
                voice=voice_id,
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

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
            reference_audio: Path to reference audio (3-10 seconds recommended)
            text: Text to synthesize with cloned voice
            reference_text: Transcript of reference audio (optional but improves quality)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Load reference audio if needed
            if isinstance(reference_audio, np.ndarray):
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, reference_audio, 16000)
                    reference_audio = f.name

            # Use zero-shot or cross-lingual based on whether we have reference text
            if reference_text:
                # Zero-shot with known prompt text
                generator = self.model.inference_zero_shot(
                    text,
                    reference_text,
                    reference_audio,
                )
            else:
                # Cross-lingual (doesn't need reference text)
                generator = self.model.inference_cross_lingual(
                    text,
                    reference_audio,
                )

            # Collect audio
            audio_segments = []
            sample_rate = 22050

            for segment in generator:
                if isinstance(segment, dict) and 'tts_speech' in segment:
                    audio = segment['tts_speech'].numpy()
                    audio_segments.append(audio)
                elif isinstance(segment, np.ndarray):
                    audio_segments.append(segment)

            if audio_segments:
                audio = np.concatenate(audio_segments)
            else:
                audio = np.array([], dtype=np.float32)

            return AudioResult(
                audio=audio.flatten(),
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "cosyvoice",
                voice="cloned",
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Voice cloning failed: {e}")

    def synthesize_with_instruct(
        self,
        text: str,
        voice: str,
        instruct: str,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize with instruction-following (requires instruct model).

        Args:
            text: Text to synthesize
            voice: Voice ID
            instruct: Instruction like "用轻柔的语气说" (speak gently)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            generator = self.model.inference_instruct(text, voice, instruct)

            audio_segments = []
            sample_rate = 22050

            for segment in generator:
                if isinstance(segment, dict) and 'tts_speech' in segment:
                    audio_segments.append(segment['tts_speech'].numpy())

            audio = np.concatenate(audio_segments) if audio_segments else np.array([])

            return AudioResult(
                audio=audio.flatten(),
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "cosyvoice",
                voice=voice,
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Instruct synthesis failed: {e}")

    def list_voices(self) -> List[VoiceInfo]:
        """List available built-in voices."""
        voices = []
        for voice_id, info in self.BUILTIN_VOICES.items():
            voices.append(VoiceInfo(
                id=voice_id,
                name=info["description"],
                language=info["language"],
                gender=info["gender"],
                sample_rate=22050,
            ))
        return voices

    def _adjust_speed(self, audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
        """Adjust audio playback speed."""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except:
            return audio

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return ["zh", "en", "ja", "ko", "yue"]

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return True

    @classmethod
    def supports_streaming(cls) -> bool:
        return True
