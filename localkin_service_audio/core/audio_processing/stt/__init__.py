"""
Speech-to-Text strategies.
"""
from .base import STTStrategy
from .whisper_strategy import WhisperStrategy
from .faster_whisper_strategy import FasterWhisperStrategy
from .whisper_cpp_strategy import WhisperCppStrategy
from .sensevoice_strategy import SenseVoiceStrategy
from .paraformer_strategy import ParaformerStrategy
from .moonshine_strategy import MoonshineStrategy

__all__ = [
    "STTStrategy",
    # Whisper family
    "WhisperStrategy",
    "FasterWhisperStrategy",
    "WhisperCppStrategy",
    # Chinese / Multilingual
    "SenseVoiceStrategy",
    "ParaformerStrategy",
    # Fast English
    "MoonshineStrategy",
]
