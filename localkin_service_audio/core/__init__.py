"""
Core functionality for LocalKin Service Audio.

This module contains the core components including configuration management,
model handling, and audio processing.
"""

from .config import get_models, find_model, find_models_by_type, validate_model_config, save_models_config, get_config_metadata
from .models import (
    list_local_models, pull_model, pull_ollama_model, pull_huggingface_model,
    get_cache_info, clear_cache, run_ollama_model, run_huggingface_model
)
from .audio_processing.stt import transcribe_audio
from .audio_processing.tts import synthesize_speech

__all__ = [
    # Config
    "get_models", "find_model", "find_models_by_type", "validate_model_config", "save_models_config", "get_config_metadata",
    # Models
    "list_local_models", "pull_model", "pull_ollama_model", "pull_huggingface_model",
    "get_cache_info", "clear_cache", "run_ollama_model", "run_huggingface_model",
    # Audio Processing
    "transcribe_audio", "synthesize_speech"
]
