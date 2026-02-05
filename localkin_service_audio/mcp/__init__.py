"""
MCP (Model Context Protocol) server for LocalKin Audio.

Enables Claude Code/Desktop integration for audio transcription and synthesis.
"""
from .server import create_server, run_server

__all__ = ["create_server", "run_server"]
