"""
Command-line interface for LocalKin Service Audio.

This module contains the CLI application and command definitions.

New CLI (v2.0) uses Click for better structure and help.
Legacy CLI (v1.x) uses argparse and is available for backward compatibility.
"""
# New Click-based CLI (v2.0)
from .main import cli, main

# Legacy argparse CLI (v1.x) - for backward compatibility
from .cli_legacy import main as legacy_main

__all__ = ["cli", "main", "legacy_main"]
