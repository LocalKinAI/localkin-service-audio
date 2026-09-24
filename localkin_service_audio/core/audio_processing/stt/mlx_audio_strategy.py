"""
mlx-audio STT Strategy - every speech recognizer mlx-audio ships, on Apple Silicon.

One strategy covers Qwen3-ASR, FireRedASR2, Fun-ASR-Nano, Parakeet, Nemotron
streaming ASR, Voxtral Realtime, VibeVoice-ASR, GLM-ASR, MOSS-Transcribe-
Diarize and the rest: the registry entry names the repo, mlx-audio picks the
architecture from its config.
"""
import os
import tempfile
import time
from typing import List, Optional, Union

import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig
from ..mlx_audio_support import require_mlx_audio
from ..shared import LANGUAGE_NAMES, audio_duration, call_supported, segments_from, to_iso


class MLXAudioSTTStrategy(STTStrategy):
    """
    Speech-to-Text through mlx-audio.

    Registry ``parameters`` understood here:
        language_names: pass "Chinese" rather than "zh" (Qwen3-ASR family)
        defaults:       extra ``generate`` kwargs, e.g. {"max_tokens": 8192}
    """

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        repo = model_config.local_path or model_config.repo_id
        try:
            require_mlx_audio()
            if not repo:
                raise ValueError(f"{model_config.name} has no repo_id")
            from mlx_audio.stt.utils import load

            print(f"Loading {model_config.name} ({repo}) with mlx-audio...")
            self.model = load(repo)
            self.model_config = model_config
            self.device = "mlx"
            self._is_loaded = True
            print(f"{model_config.name} loaded")
            return True
        except Exception as e:
            self.load_error = str(e)
            print(f"Failed to load {model_config.name}: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()
        params = dict(self.model_config.parameters or {})
        gen_kwargs = {**params.get("defaults", {}), **kwargs}
        if language and language != "auto":
            gen_kwargs["language"] = (
                LANGUAGE_NAMES.get(language, language)
                if params.get("language_names") else language
            )

        # Models differ on array input (sample rate, mx vs numpy); a path is
        # the one form every family accepts.
        temp_path = None
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(temp_path, audio, 16000)
            audio_path = temp_path
        else:
            audio_path = audio

        try:
            out = call_supported(self.model.generate, audio_path, verbose=False, **gen_kwargs)
            duration = audio_duration(audio_path)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            if temp_path:
                os.unlink(temp_path)

        # Some families stream by default and hand back a generator.
        if not hasattr(out, "text") and hasattr(out, "__iter__"):
            chunks = list(out)
            out = chunks[-1] if chunks and hasattr(chunks[-1], "text") else None
            text = "".join(str(getattr(c, "text", c)) for c in chunks) if out is None else out.text
        else:
            text = out.text

        detected = getattr(out, "language", None) if out is not None else None
        if isinstance(detected, (list, tuple)):
            detected = detected[0] if detected else None

        return TranscriptionResult(
            text=(text or "").strip(),
            language=to_iso(detected) or language,
            duration=duration,
            segments=segments_from(out),
            model=self.model_config.name,
            engine="mlx-audio",
            processing_time=time.time() - start_time,
        )

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return sorted(LANGUAGE_NAMES)
