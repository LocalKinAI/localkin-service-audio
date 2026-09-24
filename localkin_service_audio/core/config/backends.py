"""
Backend selection for models that can run more than one way.

A catalog entry such as ``qwen3-asr:1.7b`` carries an ``mlx`` backend
(mlx-audio, Apple Silicon) and a ``torch`` backend (transformers or the
model's official package; CUDA, CPU, MPS). Callers use one name everywhere;
the backend is picked per machine:

1. ``LOCALKIN_BACKEND=mlx|torch`` forces one (handy for testing the torch
   path on a Mac).
2. Otherwise mlx when on Apple Silicon with mlx-audio installed — it is
   usually the fastest option there — else torch.
3. If the preferred backend isn't defined for the model, whichever is.
"""
import dataclasses
import importlib.util
import os
from typing import List, Optional

from ..types import ModelConfig

BACKEND_ENV = "LOCALKIN_BACKEND"


def mlx_available() -> bool:
    from ..audio_processing.mlx_audio_support import is_apple_silicon

    return is_apple_silicon() and importlib.util.find_spec("mlx_audio") is not None


def backend_order() -> List[str]:
    forced = os.environ.get(BACKEND_ENV, "").strip().lower()
    if forced:
        return [forced]
    return ["mlx", "torch"] if mlx_available() else ["torch", "mlx"]


def resolve_backend(config: Optional[ModelConfig]) -> Optional[ModelConfig]:
    """Return ``config`` with its chosen backend's fields applied.

    Single-backend configs pass through unchanged.
    """
    if config is None or not config.backends:
        return config
    order = backend_order()
    choice = next((b for b in order if b in config.backends), None)
    if choice is None:
        # Forced to a backend this model lacks: use what it has, so the
        # error comes from that backend's loader with a useful message.
        choice = next(iter(config.backends))
    override = dict(config.backends[choice])
    return dataclasses.replace(config, backends={}, backend=choice, **override)
