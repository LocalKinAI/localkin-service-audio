"""
Audio processing functionality for LocalKin Service Audio.

This module contains speech-to-text and text-to-speech processing.
"""

from .stt import transcribe_audio
from .tts import synthesize_speech

__all__ = ["transcribe_audio", "synthesize_speech"]
