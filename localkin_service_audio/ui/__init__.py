"""
Web User Interface for LocalKin Service Audio.

This module provides a modern web-based interface for audio processing,
built on top of FastAPI with HTML templates and interactive features.
"""

from .routes import create_ui_router

__all__ = ["create_ui_router"]
