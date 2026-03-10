"""
Core functionality for LocalKin Service Audio.

This module contains the core components including configuration management,
model handling, and audio processing.

API:
    from localkin_service_audio.core import AudioEngine, model_registry, transcribe, synthesize
"""
# Types
from .types import (
    ModelType,
    DeviceType,
    Segment,
    TranscriptionResult,
    AudioResult,
    VoiceInfo,
    ModelConfig,
    HardwareRequirements,
    HardwareProfile,
)
# Audio Engine
from .audio_processing import (
    AudioEngine,
    get_audio_engine,
    transcribe,
    synthesize,
    STTStrategy,
    TTSStrategy,
)
# Config
from .config import (
    ModelRegistry,
    model_registry,
    Settings,
    settings,
)
# Model management (cache, pull, etc.)
from .models import (
    list_local_models, pull_model, pull_ollama_model, pull_huggingface_model,
    get_cache_info, clear_cache, run_ollama_model, run_huggingface_model
)

__all__ = [
    # Types
    "ModelType", "DeviceType", "Segment", "TranscriptionResult", "AudioResult",
    "VoiceInfo", "ModelConfig", "HardwareRequirements", "HardwareProfile",
    # Engine
    "AudioEngine", "get_audio_engine", "transcribe", "synthesize",
    "STTStrategy", "TTSStrategy",
    # Registry
    "ModelRegistry", "model_registry", "Settings", "settings",
    # Models
    "list_local_models", "pull_model", "pull_ollama_model", "pull_huggingface_model",
    "get_cache_info", "clear_cache", "run_ollama_model", "run_huggingface_model",
]
