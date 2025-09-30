"""
LocalKin Service Audio - Local Speech-to-Text and Text-to-Speech Model Manager

A CLI tool for managing and running local STT and TTS models,
inspired by Ollama's simplicity for local AI model management.
"""

__version__ = "1.0.5"
__author__ = "LocalKin Team"
__description__ = "Local STT & TTS Model Manager"

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
    "get_model_template", "list_available_templates", "create_model_from_template"
]
