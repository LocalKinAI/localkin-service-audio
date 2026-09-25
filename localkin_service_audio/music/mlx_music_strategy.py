"""
Music with mlx-audio on Apple Silicon (MiniMax Music 3).

Much faster on a Mac than the same model through ComfyUI: there the text
encoder ships int8-"convrot" weights built for CUDA, and on MPS its
autoregressive stage ran at 3-9 s per step — about an hour for a 30-second
song. mlx-audio runs it natively.
"""
import time
from typing import Any, Dict, Optional

import numpy as np

from .base import MusicEngine
from ..core.audio_processing.mlx_audio_support import is_apple_silicon, require_mlx_audio
from ..core.types import AudioResult, ModelConfig

# kin model name -> mlx-community repo
MODELS = {
    "minimax-music3": "mlx-community/MiniMax-Music3-mxfp8",
}


def mlx_music_available(model: str) -> bool:
    import importlib.util

    return model in MODELS and is_apple_silicon() and importlib.util.find_spec("mlx_audio") is not None


class MLXMusicStrategy(MusicEngine):
    def __init__(self):
        super().__init__()
        self.load_error: Optional[str] = None

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        repo = model_config.repo_id or MODELS.get(model_config.name)
        try:
            require_mlx_audio()
            if not repo:
                raise ValueError(f"no MLX build known for {model_config.name}")
            from mlx_audio.music.utils import load

            print(f"Loading {model_config.name} ({repo}) with mlx-audio...")
            self.model = load(repo)
        except Exception as e:
            self.load_error = str(e)
            print(f"Failed to load {model_config.name}: {e}")
            return False
        self.model_config = model_config
        self.device = "mlx"
        self._is_loaded = True
        return True

    def generate(
        self,
        prompt: str,
        duration: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        lyrics: Optional[str] = None,
        seed: Optional[int] = None,
        tags: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AudioResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        start = time.time()
        text = f"{prompt}, {tags}" if tags else prompt
        # The model wants lyrics; an explicit tag asks for an instrumental.
        results = self.model.generate(
            text,
            lyrics=lyrics if lyrics and lyrics.strip() else "[instrumental]",
            duration=float(duration) if duration else 60.0,
            seed=seed if seed is not None else int(time.time()) % 2**31,
            **(params or {}),
        )
        chunks, sample_rate = [], 44100
        for r in results:
            chunks.append(np.asarray(r.audio, dtype=np.float32))
            sample_rate = r.sample_rate
        audio = np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)
        return AudioResult(audio=audio, sample_rate=int(sample_rate), model=self.model_config.name,
                           duration=len(audio) / sample_rate, processing_time=time.time() - start)

    def get_info(self) -> Dict[str, Any]:
        return {"engine": "MLXMusicStrategy", "device": self.device,
                "repo": (self.model_config.repo_id if self.model_config else None)}
