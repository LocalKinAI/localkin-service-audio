"""
Music Generation Module for LocalKin Service Audio.

Provides music generation capabilities through pluggable music engines.

Supported engines:
- MusicGenStrategy: Meta's MusicGen model (facebook/musicgen-{small|medium|large})
- HeartMuLaStrategy: HeartMuLa open-source model (Chinese support, multilingual)

Example:
    from localkin_service_audio.music import MusicEngine, MusicGenStrategy, HeartMuLaStrategy
    from localkin_service_audio.core.types import ModelConfig

    # Load MusicGen model
    config = ModelConfig(
        name="musicgen-small",
        type=ModelType.TTS,
    )
    engine = MusicGenStrategy()
    engine.load(config, device="auto")
    result = engine.generate("calm piano melody", duration=10)
    result.save("music.wav")

    # Load HeartMuLa model (with Chinese support)
    config = ModelConfig(name="heartmula:3b")
    engine = HeartMuLaStrategy()
    engine.load(config, device="auto")
    result = engine.generate(
        "在月光下弹钢琴",  # Chinese lyrics
        tags="piano,romantic",
        duration=30
    )
    result.save("music.wav")
"""

from .base import MusicEngine
from .musicgen_strategy import MusicGenStrategy
from .heartmula_strategy import HeartMuLaStrategy

__all__ = [
    "MusicEngine",
    "MusicGenStrategy",
    "HeartMuLaStrategy",
]
