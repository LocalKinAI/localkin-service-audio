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

    # Language code mapping: voice prefix -> Kokoro lang_code
    LANG_CODES = {
        "a": "a",   # American English
        "b": "b",   # British English
        "e": "e",   # European (Spanish)
        "f": "f",   # French
        "h": "h",   # Hindi
        "i": "i",   # Italian
        "j": "j",   # Japanese
        "p": "p",   # Portuguese (Brazilian)
        "z": "z",   # Chinese (Mandarin)
    }

    # Kokoro voice presets
    VOICES = {
        # American English
        "af_alloy": "Alloy (American Female)",
        "af_aoede": "Aoede (American Female)",
        "af_bella": "Bella (American Female)",
        "af_heart": "Heart (American Female)",
        "af_jessica": "Jessica (American Female)",
        "af_kore": "Kore (American Female)",
        "af_nicole": "Nicole (American Female)",
        "af_nova": "Nova (American Female)",
        "af_river": "River (American Female)",
        "af_sarah": "Sarah (American Female)",
        "af_sky": "Sky (American Female)",
        "am_adam": "Adam (American Male)",
        "am_echo": "Echo (American Male)",
        "am_eric": "Eric (American Male)",
        "am_fenrir": "Fenrir (American Male)",
        "am_liam": "Liam (American Male)",
        "am_michael": "Michael (American Male)",
        "am_onyx": "Onyx (American Male)",
        "am_puck": "Puck (American Male)",
        # British English
        "bf_alice": "Alice (British Female)",
        "bf_emma": "Emma (British Female)",
        "bf_isabella": "Isabella (British Female)",
        "bf_lily": "Lily (British Female)",
        "bm_daniel": "Daniel (British Male)",
        "bm_fable": "Fable (British Male)",
        "bm_george": "George (British Male)",
        "bm_lewis": "Lewis (British Male)",
        # European (Spanish)
        "ef_dora": "Dora (European Female)",
        "em_alex": "Alex (European Male)",
        # French
        "ff_siwis": "Siwis (French Female)",
        # Hindi
        "hf_alpha": "Alpha (Hindi Female)",
        "hf_beta": "Beta (Hindi Female)",
        "hm_omega": "Omega (Hindi Male)",
        "hm_psi": "Psi (Hindi Male)",
        # Italian
        "if_sara": "Sara (Italian Female)",
        "im_nicola": "Nicola (Italian Male)",
        # Japanese
        "jf_alpha": "Alpha (Japanese Female)",
        "jf_gongitsune": "Gongitsune (Japanese Female)",
        "jf_nezumi": "Nezumi (Japanese Female)",
        "jf_tebukuro": "Tebukuro (Japanese Female)",
        "jm_kumo": "Kumo (Japanese Male)",
        # Portuguese (Brazilian)
        "pf_dora": "Dora (Portuguese Female)",
        "pm_alex": "Alex (Portuguese Male)",
        # Chinese (Mandarin)
        "zf_xiaobei": "Xiaobei (Chinese Female)",
        "zf_xiaoni": "Xiaoni (Chinese Female)",
        "zf_xiaoxiao": "Xiaoxiao (Chinese Female)",
        "zf_xiaoyi": "Xiaoyi (Chinese Female)",
        "zm_yunjian": "Yunjian (Chinese Male)",
        "zm_yunxi": "Yunxi (Chinese Male)",
        "zm_yunxia": "Yunxia (Chinese Male)",
        "zm_yunyang": "Yunyang (Chinese Male)",
    }

    @staticmethod
    def _ensure_spacy_model():
        """Ensure the spacy English model is available (needed by misaki/kokoro)."""
        try:
            import spacy.util
            if not spacy.util.is_package("en_core_web_sm"):
                print("Downloading spacy English model (one-time setup)...")
                import subprocess, sys
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    stdout=subprocess.DEVNULL,
                )
        except Exception:
            pass  # Let kokoro handle the error downstream

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load Kokoro TTS model."""
        try:
            import kokoro

            self.device = self._detect_device(device)
            model_name = model_config.model_size or "kokoro-82m"

            print(f"Loading Kokoro TTS model '{model_name}'...")

            # Ensure spacy model is available (misaki uses spacy.cli.download
            # which calls pip internally — fails in uv venvs without pip)
            self._ensure_spacy_model()

            # Initialize Kokoro pipeline for American English by default
            self._pipelines = {}
            self._pipelines["a"] = kokoro.KPipeline(lang_code='a')
            self.model = self._pipelines["a"]

            self.model_config = model_config
            self._is_loaded = True
            self._current_voice = "af_heart"  # Default voice

            print(f"Kokoro TTS loaded")
            return True

        except ImportError as e:
            err = str(e)
            if "kokoro" in err:
                print("Kokoro not installed. Install with: pip install kokoro")
            elif "AlbertModel" in err:
                print("Incompatible transformers version. Fix with: pip install 'transformers>=4.21,<4.50'")
            else:
                print(f"Kokoro dependency error: {e}")
                print("Try: pip install kokoro>=0.9.2")
            return False
        except Exception as e:
            print(f"Failed to load Kokoro model: {e}")
            return False

    def _get_pipeline(self, voice_id: str):
        """Get or create a Kokoro pipeline for the voice's language."""
        import kokoro
        lang_prefix = voice_id[0] if voice_id else "a"
        lang_code = self.LANG_CODES.get(lang_prefix, "a")
        if lang_code not in self._pipelines:
            self._pipelines[lang_code] = kokoro.KPipeline(lang_code=lang_code)
        return self._pipelines[lang_code]

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
            voice_id = voice or self._current_voice or "af_heart"

            # Get the right pipeline for this voice's language
            pipeline = self._get_pipeline(voice_id)

            # Generate audio
            generator = pipeline(
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

    # Voice prefix -> ISO language code
    VOICE_LANG_MAP = {
        "a": "en", "b": "en", "e": "es", "f": "fr",
        "h": "hi", "i": "it", "j": "ja", "p": "pt", "z": "zh",
    }

    def list_voices(self) -> List[VoiceInfo]:
        """List available Kokoro voices."""
        voices = []
        for voice_id, name in self.VOICES.items():
            lang = self.VOICE_LANG_MAP.get(voice_id[0], "en")
            gender = "female" if voice_id[1] == "f" else "male"

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
        return ["en", "es", "fr", "hi", "it", "ja", "pt", "zh"]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return cls.AVAILABLE_MODELS

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return False

    @classmethod
    def supports_streaming(cls) -> bool:
        return True  # Kokoro yields audio segments
