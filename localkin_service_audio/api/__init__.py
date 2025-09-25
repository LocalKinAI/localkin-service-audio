"""
API server functionality for LocalKin Service Audio.

This module contains the FastAPI server and endpoint definitions.
"""

from .server import create_app, run_server

__all__ = ["create_app", "run_server"]
