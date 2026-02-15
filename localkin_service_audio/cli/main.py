"""
Main CLI entry point for LocalKin Audio.

This module provides the Click-based CLI interface.
"""
import click
from typing import Optional

from .. import __version__
from .commands import (
    transcribe,
    tts,
    models,
    pull,
    rm,
    recommend,
    benchmark,
    listen,
    config,
    status,
    cache,
    ps,
    add_model,
    list_templates,
    music,
)
from .commands.serve import serve, web


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, "-V", "--version", prog_name="kin")
@click.pass_context
def cli(ctx):
    """
    🎵 LocalKin Audio - Voice AI Platform

    Local Speech-to-Text (STT) and Text-to-Speech (TTS) with
    support for multiple engines and Chinese language models.

    Examples:

        kin audio transcribe audio.wav

        kin audio tts "Hello world"

        kin audio models

        kin audio recommend
    """
    ctx.ensure_object(dict)


@cli.group()
def audio():
    """
    Audio processing commands.

    Speech-to-Text (STT) and Text-to-Speech (TTS) operations.
    """
    pass


# Register audio commands
audio.add_command(transcribe)
audio.add_command(tts)
audio.add_command(listen)
audio.add_command(models)
audio.add_command(pull)
audio.add_command(rm)
audio.add_command(serve)
audio.add_command(recommend)
audio.add_command(benchmark)
audio.add_command(config)
audio.add_command(status)
audio.add_command(cache)
audio.add_command(ps)
audio.add_command(add_model)
audio.add_command(list_templates)
audio.add_command(music)

# Top-level commands
cli.add_command(web)


@cli.command()
def mcp():
    """
    Start MCP server for Claude integration.

    Enables Claude Code/Desktop to use audio transcription and synthesis.
    """
    from .utils import print_header, print_info, print_error

    print_header("MCP Server")
    print_info("Starting Model Context Protocol server...")

    try:
        from ..mcp.server import run_server
        run_server()
    except ImportError:
        print_error("MCP server not available. Install with: pip install mcp")
    except Exception as e:
        print_error(f"MCP server failed: {e}")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed info")
def info(verbose: bool):
    """
    Show system information and loaded models.
    """
    from .utils import print_header, print_info
    from .utils.device import detect_hardware, print_hardware_info

    print_header("System Information")

    print(f"\nVersion: {__version__}")

    if verbose:
        hardware = detect_hardware()
        print_hardware_info(hardware)

    # Show loaded models
    try:
        from ..core import get_audio_engine
        engine = get_audio_engine()
        info = engine.get_info()

        print("\n📦 Loaded Models:")
        if info["stt"]["loaded"]:
            print(f"  STT: {info['stt']['model']}")
        else:
            print("  STT: (none)")

        if info["tts"]["loaded"]:
            print(f"  TTS: {info['tts']['model']}")
        else:
            print("  TTS: (none)")

    except Exception as e:
        if verbose:
            print_info(f"Could not get engine info: {e}")


def main():
    """Entry point for the CLI."""
    # Try the new Click CLI first
    try:
        cli()
    except click.exceptions.UsageError:
        # Fall back to legacy argparse CLI for backward compatibility
        from .cli import main as legacy_main
        legacy_main()


if __name__ == "__main__":
    main()
