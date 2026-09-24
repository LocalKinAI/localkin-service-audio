"""
mlx-audio specifics: platform check and mx.array conversion.

mlx-audio (https://github.com/Blaizzy/mlx-audio) runs several dozen STT and
TTS families on Apple Silicon behind one loader, so a single strategy per
direction covers them all. The generic helpers they use (language handling,
kwarg filtering, reference voices) live in shared.py.
"""
import platform
import sys
from typing import Any

import numpy as np


INSTALL_HINT = "pip install 'localkin-service-audio[mlx]'  (Apple Silicon only)"


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def require_mlx_audio() -> None:
    """Raise a clear error when mlx-audio can't run here."""
    if not is_apple_silicon():
        raise RuntimeError(
            "This model runs on mlx-audio, which needs an Apple Silicon Mac "
            f"(this is {sys.platform}/{platform.machine()})."
        )
    try:
        import mlx_audio  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"mlx-audio is not installed. Install with: {INSTALL_HINT}") from e


def to_numpy(audio: Any) -> np.ndarray:
    """mx.array (or anything array-like) -> float32 numpy, singleton dims dropped."""
    if type(audio).__module__.startswith("mlx"):
        import mlx.core as mx
        # numpy has no bfloat16, so convert on the mlx side first.
        audio = audio.astype(mx.float32)
    return np.squeeze(np.asarray(audio, dtype=np.float32))
