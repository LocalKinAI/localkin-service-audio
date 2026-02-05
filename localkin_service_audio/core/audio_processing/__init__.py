"""
Audio Processing Module for LocalKin Service Audio.

Provides unified STT (Speech-to-Text) and TTS (Text-to-Speech) capabilities
through the AudioEngine facade and pluggable strategies.

New API (v2.0):
    from localkin_service_audio.core.audio_processing import AudioEngine, transcribe, synthesize

    engine = AudioEngine()
    engine.load_stt("whisper-cpp:base")
    result = engine.transcribe("audio.wav")

Legacy API (v1.x, deprecated):
    from localkin_service_audio.core.audio_processing import transcribe_audio, synthesize_speech
"""
from .engine import (
    AudioEngine,
    get_audio_engine,
    transcribe,
    synthesize,
)
from .stt import STTStrategy
from .tts import TTSStrategy

# Legacy imports for backward compatibility
from .stt_legacy import (
    transcribe_audio,
    get_available_engines,
    get_faster_whisper_models,
)
from .tts_legacy import synthesize_speech

__all__ = [
    # New API (v2.0)
    "AudioEngine",
    "get_audio_engine",
    "transcribe",
    "synthesize",
    "STTStrategy",
    "TTSStrategy",
    # Legacy API (v1.x, deprecated)
    "transcribe_audio",
    "synthesize_speech",
    "get_available_engines",
    "get_faster_whisper_models",
]
