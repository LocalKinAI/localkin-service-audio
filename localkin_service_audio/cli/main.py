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
@click.option("--stt", help="Default speech-to-text model (qwen3-asr, sized to this machine)")
@click.option("--tts", help="Default text-to-speech model (qwen3-tts:0.6b)")
@click.option("--clone", "clone_model", help="Default voice-cloning model (qwen3-tts base)")
@click.option("--music", "music_model", help="Default music model (musicgen:small)")
@click.option("--http", "http", is_flag=True,
              help="Serve over HTTP (at /mcp) for clients on other machines, instead of stdio")
@click.option("--host", default="127.0.0.1", show_default=True, help="With --http: address to listen on")
@click.option("--port", default=8765, show_default=True, help="With --http: port")
@click.option("--api-key", envvar="LOCALKIN_API_KEY", default=None,
              help="With --http: require this key (Authorization: Bearer). Env: LOCALKIN_API_KEY")
@click.option("--public-url", default=None,
              help="With --http: the address clients use, for download links (default: from --host/--port)")
def mcp(stt, tts, clone_model, music_model, http, host, port, api_key, public_url):
    """
    MCP server for Claude Code, Claude Desktop, Cursor, Codex and other MCP
    clients. On this machine the client starts it over stdio:

    \b
        claude mcp add localkin-audio -- kin mcp

    For clients elsewhere (another computer, a cloud agent), run it over HTTP
    and point them at http://<this machine>:8765/mcp:

    \b
        kin mcp --http --host 0.0.0.0 --api-key SECRET

    Tools: transcribe_audio, synthesize_speech, clone_voice, generate_music,
    list_models, list_voices, get_audio_info.
    """
    if http:
        from ..mcp.server import run_http_server

        run_http_server(host=host, port=port, api_key=api_key, public_url=public_url,
                        stt=stt, tts=tts, clone=clone_model, music=music_model)
        return
    # Nothing may be printed here: stdout belongs to the protocol.
    from ..mcp.server import run_server

    run_server(stt=stt, tts=tts, clone=clone_model, music=music_model)


@cli.command()
@click.option("--dir", "target", default=None,
              help="Skills directory to install into (default: ~/.claude/skills)")
@click.option("--print", "show", is_flag=True, help="Print the skill instead of installing it")
def skill(target, show):
    """
    Install the agent skill that teaches an agent to use kin: which model
    for which job, and the commands for transcribing, speaking, cloning,
    music and serving.

    \b
        kin skill                               # Claude Code, every project
        kin skill --dir .claude/skills          # Claude Code, this project only
        kin skill --dir ~/.openclaw/skills      # another agent's skills folder
        kin skill --print                       # the text, to paste anywhere
    """
    from importlib import resources
    from pathlib import Path

    from .utils import print_success

    text = (resources.files("localkin_service_audio") / "skills" / "localkin-audio" / "SKILL.md").read_text(
        encoding="utf-8")
    if show:
        click.echo(text, nl=False)
        return
    dest = Path(target).expanduser() if target else Path.home() / ".claude" / "skills"
    path = dest / "localkin-audio" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print_success(f"Installed {path}")


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
    cli()


if __name__ == "__main__":
    main()
