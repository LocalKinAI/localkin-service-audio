"""
Core functionality for LocalKin Service Audio.

This module contains the core components including configuration management,
model handling, and audio processing.

New API (v2.0):
    from localkin_service_audio.core import AudioEngine, model_registry, transcribe, synthesize

Legacy API (v1.x):
    from localkin_service_audio.core import transcribe_audio, synthesize_speech
"""
# New API (v2.0)
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
from .audio_processing import (
    AudioEngine,
    get_audio_engine,
    transcribe,
    synthesize,
    STTStrategy,
    TTSStrategy,
)
from .config import (
    ModelRegistry,
    model_registry,
    Settings,
    settings,
)

# Legacy API (v1.x) - for backward compatibility
from .config_legacy import get_models, find_model, find_models_by_type, validate_model_config, save_models_config, get_config_metadata
from .models import (
    list_local_models, pull_model, pull_ollama_model, pull_huggingface_model,
    get_cache_info, clear_cache, run_ollama_model, run_huggingface_model
)
from .audio_processing import transcribe_audio, synthesize_speech

__all__ = [
    # New API - Types
    "ModelType", "DeviceType", "Segment", "TranscriptionResult", "AudioResult",
    "VoiceInfo", "ModelConfig", "HardwareRequirements", "HardwareProfile",
    # New API - Engine
    "AudioEngine", "get_audio_engine", "transcribe", "synthesize",
    "STTStrategy", "TTSStrategy",
    # New API - Registry
    "ModelRegistry", "model_registry", "Settings", "settings",
    # Legacy API - Config
    "get_models", "find_model", "find_models_by_type", "validate_model_config", "save_models_config", "get_config_metadata",
    # Legacy API - Models
    "list_local_models", "pull_model", "pull_ollama_model", "pull_huggingface_model",
    "get_cache_info", "clear_cache", "run_ollama_model", "run_huggingface_model",
    # Legacy API - Audio Processing
    "transcribe_audio", "synthesize_speech"
]
