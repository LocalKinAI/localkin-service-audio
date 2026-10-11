"""
MCP (Model Context Protocol) server for LocalKin Audio.

Speech-to-text, text-to-speech, voice cloning and music for any MCP client.
"""
from .server import AudioTools, create_server, default_models, run_server

__all__ = ["AudioTools", "create_server", "default_models", "run_server"]
