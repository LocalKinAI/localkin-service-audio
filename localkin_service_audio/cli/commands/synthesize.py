"""
Synthesize command - Text-to-Speech operations.
"""
import click
import os
import time
from typing import Optional

from ..utils import print_success, print_error, print_info, print_header


@click.command("tts")
@click.argument("text")
@click.option(
    "--model", "-m",
    default="kokoro",
    help="Model to use (e.g., kokoro, cosyvoice:300m, chattts, native)"
)
@click.option(
    "--voice", "-v",
    default=None,
    help="Voice ID to use. Use --list-voices to see available voices."
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path. If not specified, plays audio directly."
)
@click.option(
    "--speed", "-s",
    type=float,
    default=1.0,
    help="Speech speed multiplier (0.5 = half speed, 2.0 = double speed)"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for inference."
)
@click.option(
    "--list-voices",
    is_flag=True,
    help="List available voices for the model and exit."
)
@click.option(
    "--play/--no-play",
    default=True,
    help="Play audio after synthesis (when output is specified)."
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output."
)
def tts(
    text: str,
    model: str,
    voice: Optional[str],
    output: Optional[str],
    speed: float,
    device: str,
    list_voices: bool,
    play: bool,
    verbose: bool
):
    """
    Synthesize speech from text.

    Examples:

        kin audio tts "Hello, world!"

        kin audio tts "你好世界" --model cosyvoice:300m --voice 中文女

        kin audio tts "Hello" --output hello.wav --model kokoro --voice af_bella

        kin audio tts --model kokoro --list-voices
    """
    if verbose:
        print_header("Text-to-Speech")

    try:
        from ...core import get_audio_engine

        engine = get_audio_engine()

        # Load model
        if verbose:
            print_info(f"Loading model: {model}")

        if not engine._tts_strategy or engine._tts_model_name != model:
            success = engine.load_tts(model, device=device)
            if not success:
                print_error(f"Failed to load model: {model}")
                return

        # List voices if requested
        if list_voices:
            voices = engine.list_voices()
            if voices:
                print("\nAvailable voices:")
                print("-" * 50)
                for v in voices:
                    gender = f" ({v.gender})" if v.gender else ""
                    print(f"  {v.id:<20} {v.name}{gender}")
            else:
                print_info("No voice information available for this model.")
            return

        # Synthesize
        if verbose:
            print_info("Synthesizing...")

        start_time = time.time()
        result = engine.synthesize(text, voice=voice, speed=speed)
        elapsed = time.time() - start_time

        # Output
        if output:
            result.save(output)
            print_success(f"Saved to {output}")

            if play:
                _play_audio(output)
        else:
            # Save to temp file and play
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            result.save(temp_path)
            _play_audio(temp_path)
            os.unlink(temp_path)

        if verbose:
            print_info(f"Synthesis completed in {elapsed:.2f}s")
            print_info(f"Duration: {result.duration:.2f}s")

    except Exception as e:
        print_error(f"Synthesis failed: {e}")
        raise click.Abort()


# Alias
synthesize = tts


def _play_audio(audio_path: str):
    """Play audio file using available system player."""
    import platform
    import subprocess

    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_path], check=True)
        elif system == "Linux":
            # Try various Linux audio players
            for player in ["aplay", "paplay", "play"]:
                try:
                    subprocess.run([player, audio_path], check=True)
                    break
                except FileNotFoundError:
                    continue
        elif system == "Windows":
            import winsound
            winsound.PlaySound(audio_path, winsound.SND_FILENAME)
    except Exception as e:
        print_info(f"Could not play audio: {e}")
