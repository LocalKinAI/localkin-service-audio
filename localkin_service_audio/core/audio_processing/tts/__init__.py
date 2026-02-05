"""
Text-to-Speech strategies.
"""
from .base import TTSStrategy
from .native_strategy import NativeStrategy
from .kokoro_strategy import KokoroStrategy
from .cosyvoice_strategy import CosyVoiceStrategy
from .chattts_strategy import ChatTTSStrategy
from .f5_strategy import F5TTSStrategy

__all__ = [
    "TTSStrategy",
    # Basic
    "NativeStrategy",
    "KokoroStrategy",
    # Chinese
    "CosyVoiceStrategy",
    "ChatTTSStrategy",
    # Voice Cloning
    "F5TTSStrategy",
]
