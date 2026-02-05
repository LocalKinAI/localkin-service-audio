"""
Configuration management for LocalKin Audio.
"""
from .model_registry import ModelRegistry, model_registry
from .settings import Settings, settings

__all__ = [
    "ModelRegistry",
    "model_registry",
    "Settings",
    "settings",
]
