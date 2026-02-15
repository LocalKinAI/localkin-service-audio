"""
LocalKin Service Audio - Voice AI Platform

A comprehensive voice AI platform with:
- Multiple STT engines (Whisper, SenseVoice, Paraformer, Moonshine)
- Multiple TTS engines (Kokoro, CosyVoice, ChatTTS, F5-TTS)
- Chinese language support
- Voice cloning capabilities
- MCP integration for Claude Code/Desktop

Inspired by Ollama's simplicity for local AI model management.
"""

import os
from pathlib import Path

__version__ = "2.0.4"
__author__ = "LocalKin Team"
__description__ = "Voice AI Platform - Local STT/TTS with Chinese Language Support"

def get_sample_audio_path():
    """Get the path to the included sample audio file for testing.
    
    Returns:
        str: Absolute path to the sample.wav file
        
    Example:
        >>> import localkin_service_audio as lsa
        >>> sample_path = lsa.get_sample_audio_path()
        >>> print(f"Sample audio: {sample_path}")
    """
    package_dir = Path(__file__).parent
    sample_path = package_dir / "samples" / "sample.wav"
    return str(sample_path)

# Import main CLI entry point
from .cli import main

# Import core functionality
from .core import (
    get_models, find_model, find_models_by_type,
    list_local_models, pull_model,
    transcribe_audio, synthesize_speech
)

# Import API functionality
from .api import create_app, run_server

# Import UI functionality
from .ui import create_ui_router

# Import templates
from .templates import (
    get_model_template, list_available_templates, create_model_from_template
)

# Import music functionality
from .music import (
    MusicEngine, MusicGenStrategy, HeartMuLaStrategy
)

__all__ = [
    # CLI
    "main",
    # Core functionality
    "get_models", "find_model", "find_models_by_type",
    "list_local_models", "pull_model",
    "transcribe_audio", "synthesize_speech",
    # API
    "create_app", "run_server",
    # UI
    "create_ui_router",
    # Templates
    "get_model_template", "list_available_templates", "create_model_from_template",
    # Music
    "MusicEngine", "MusicGenStrategy", "HeartMuLaStrategy",
    # Utilities
    "get_sample_audio_path"
]
