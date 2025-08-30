"""
LocalKin Service Audio - Local Speech-to-Text and Text-to-Speech Model Manager

A CLI tool for managing and running local STT and TTS models,
inspired by Ollama's simplicity for local AI model management.
"""

__version__ = "0.1.0"
__author__ = "LocalKin Team"
__description__ = "Local STT & TTS Model Manager"

from .cli import main
from .config import get_models, find_model
from .models import list_local_models, pull_model
from .stt import transcribe_audio
from .tts import synthesize_speech

__all__ = [
    "main",
    "get_models",
    "find_model",
    "list_local_models",
    "pull_model",
    "transcribe_audio",
    "synthesize_speech",
]
