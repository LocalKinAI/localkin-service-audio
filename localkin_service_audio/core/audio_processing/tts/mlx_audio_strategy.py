"""
mlx-audio TTS Strategy - every speech synthesizer mlx-audio ships, on Apple Silicon.

One strategy covers Qwen3-TTS, VoxCPM2, Chatterbox, VibeVoice, OmniVoice,
Spark, CSM, Dia, Orpheus, IndexTTS, fish-speech S2, Higgs, MOSS-TTS, Voxtral
TTS and the rest: the registry entry names the repo, mlx-audio picks the
architecture from its config.
"""
import time
from typing import List, Optional, Union

import numpy as np

from .base import TTSStrategy
from ...types import AudioResult, ModelConfig, VoiceInfo
from ..mlx_audio_support import require_mlx_audio, to_numpy
from ..shared import (
    apply_speed, call_supported, default_reference, detect_text_language, pick_voice,
    resolve_language, voice_infos,
)


class MLXAudioTTSStrategy(TTSStrategy):
    """
    Text-to-Speech through mlx-audio.

    Registry ``parameters`` understood here:
        defaults:  extra ``generate`` kwargs, e.g. {"voice": "vivian"}
        lang_arg:  name of the model's language kwarg ("lang_code"/"language")
        lang_map:  ISO code -> value for ``lang_arg``; picked from the text's
                   script when the caller gives no language
        needs_reference: clone-only model; fall back to a bundled voice
        voices:    voice ids to advertise from list_voices()
        voice_aliases: Kokoro-style prefix ("zf") -> voice, for foreign ids
    """

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        repo = model_config.local_path or model_config.repo_id
        try:
            require_mlx_audio()
            if not repo:
                raise ValueError(f"{model_config.name} has no repo_id")
            from mlx_audio.tts.utils import load_model

            print(f"Loading {model_config.name} ({repo}) with mlx-audio...")
            self.model = load_model(repo)
            self.model_config = model_config
            self.device = "mlx"
            self._is_loaded = True
            print(f"{model_config.name} loaded")
            return True
        except Exception as e:
            self.load_error = str(e)
            print(f"Failed to load {model_config.name}: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()
        params = self.model_config.parameters or {}
        language = kwargs.pop("language", None)
        gen_kwargs = {**params.get("defaults", {}), **kwargs}
        if voice:
            gen_kwargs["voice"] = pick_voice(voice, params.get("voices"), params.get("voice_aliases"),
                                             params.get("defaults", {}).get("voice"))
        lang_arg = params.get("lang_arg")
        if lang_arg and lang_arg not in gen_kwargs:
            gen_kwargs[lang_arg] = resolve_language(language, params.get("lang_map"), text)

        # Clone-only models (IndexTTS, MOSS-TTS, ...) have no built-in voice.
        # Without a caller-supplied reference they get a bundled one in the
        # text's language, so every model can answer a plain /synthesize.
        if params.get("needs_reference") and not gen_kwargs.get("ref_audio"):
            gen_kwargs["ref_audio"], gen_kwargs["ref_text"] = default_reference(
                language or detect_text_language(text)
            )

        ref_audio = gen_kwargs.get("ref_audio")
        if isinstance(ref_audio, str) and not getattr(self.model, "preserve_ref_audio_path", False):
            from mlx_audio.utils import load_audio
            gen_kwargs["ref_audio"] = load_audio(ref_audio, sample_rate=self.model.sample_rate)

        try:
            results = call_supported(self.model.generate, text=text, verbose=False, **gen_kwargs)
            chunks, sample_rate = [], getattr(self.model, "sample_rate", 24000)
            for r in results:
                chunks.append(to_numpy(r.audio))
                sample_rate = getattr(r, "sample_rate", None) or sample_rate
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

        audio = np.concatenate([c.reshape(-1) for c in chunks]) if chunks else np.array([], dtype=np.float32)
        audio = apply_speed(audio, speed, int(sample_rate))
        if audio.size == 0:
            raise RuntimeError(
                f"{self.model_config.name} produced no audio for voice "
                f"{gen_kwargs.get('voice')!r} ({len(text)} chars)"
            )

        return AudioResult(
            audio=audio,
            sample_rate=int(sample_rate),
            model=self.model_config.name,
            voice=gen_kwargs.get("voice"),
            duration=len(audio) / sample_rate,
            processing_time=time.time() - start_time,
        )

    def clone_voice(
        self,
        reference_audio: Union[str, np.ndarray],
        text: str,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> AudioResult:
        if not self.model_config.supports_voice_cloning:
            raise NotImplementedError(f"{self.model_config.name} does not support voice cloning")
        return self.synthesize(text, ref_audio=reference_audio, ref_text=reference_text, **kwargs)

    def list_voices(self) -> List[VoiceInfo]:
        params = self.model_config.parameters or {}
        lang = (self.model_config.languages or ["en"])[0]
        if params.get("voices"):
            return voice_infos(params, lang)
        voices = self.model_config.voices or []
        if not voices and self.model is not None and hasattr(self.model, "get_supported_speakers"):
            try:
                voices = list(self.model.get_supported_speakers() or [])
            except Exception:
                voices = []
        return [VoiceInfo(id=v, name=v, language=lang) for v in voices]

    @classmethod
    def supports_voice_cloning(cls) -> bool:
        return True  # per model; see ModelConfig.supports_voice_cloning
