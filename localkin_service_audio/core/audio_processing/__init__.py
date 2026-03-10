"""
Audio Processing Module for LocalKin Service Audio.

Provides unified STT (Speech-to-Text) and TTS (Text-to-Speech) capabilities
through the AudioEngine facade and pluggable strategies.

Usage:
    from localkin_service_audio.core.audio_processing import AudioEngine, transcribe, synthesize

    engine = AudioEngine()
    engine.load_stt("whisper-cpp:base")
    result = engine.transcribe("audio.wav")
"""
from .engine import (
    AudioEngine,
    get_audio_engine,
    transcribe,
    synthesize,
)
from .stt import STTStrategy
from .tts import TTSStrategy

__all__ = [
    "AudioEngine",
    "get_audio_engine",
    "transcribe",
    "synthesize",
    "STTStrategy",
    "TTSStrategy",
]
