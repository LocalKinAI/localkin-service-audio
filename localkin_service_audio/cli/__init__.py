"""
Command-line interface for LocalKin Service Audio.

This module contains the CLI application and command definitions.
Uses Click for structured commands and help.
"""
from .main import cli, main

__all__ = ["cli", "main"]
