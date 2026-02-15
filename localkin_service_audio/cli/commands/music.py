"""
Music generation command - AI music generation operations.
"""
import click
import os
import time
from typing import Optional
from pathlib import Path

from ..utils import print_success, print_error, print_info, print_header


@click.group("music")
def music():
    """
    Music generation commands.

    Generate AI music from text descriptions.

    Examples:

        kin audio music generate "calm piano melody"

        kin audio music generate "epic orchestral" --duration 15 --model medium

        kin audio music models

        kin audio music generate "ambient" -o music.wav --device mps
    """
    pass


@music.command("generate")
@click.argument("prompt")
@click.option(
    "--model", "-m",
    default="musicgen:small",
    help="Model to use (musicgen:small/medium/large, heartmula:3b/7b)"
)
@click.option(
    "--tags",
    default=None,
    help="Music style tags (for HeartMuLa: piano,happy,wedding,etc.)"
)
@click.option(
    "--duration", "-d",
    type=int,
    default=10,
    help="Duration in seconds (5-30 for MusicGen, up to 240 for HeartMuLa)"
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path. If not specified, saves to temp file and plays."
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for inference."
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Sampling temperature (higher = more creative)"
)
@click.option(
    "--play/--no-play",
    default=True,
    help="Play audio after generation (when output is specified)."
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose output."
)
def generate(
    prompt: str,
    model: str,
    tags: Optional[str],
    duration: int,
    output: Optional[str],
    device: str,
    temperature: float,
    play: bool,
    verbose: bool
):
    """
    Generate music from text prompt.

    Examples:

        # Using MusicGen
        kin audio music generate "upbeat electronic dance music"

        kin audio music generate "calm piano" --duration 20 --model musicgen:medium

        # Using HeartMuLa (supports Chinese)
        kin audio music generate "在月光下弹钢琴" --model heartmula:3b

        kin audio music generate "happy wedding" --tags "piano,romantic,wedding" --model heartmula:3b --duration 30

        kin audio music generate "ambient" -o output.wav --device mps
    """
    if verbose:
        print_header("Music Generation")
        print_info(f"Prompt: {prompt}")
        print_info(f"Model: {model}")
        if tags:
            print_info(f"Tags: {tags}")
        print_info(f"Duration: {duration}s")
        print_info(f"Device: {device}")

    try:
        from ...core.types import ModelConfig, ModelType
        from ...music import MusicGenStrategy, HeartMuLaStrategy

        # Create model config
        config = ModelConfig(
            name=model,
            type=ModelType.TTS,  # Use TTS as fallback for music generation
            engine="music"
        )

        # Select engine based on model name
        if model.startswith("heartmula") or model.startswith("heart"):
            if verbose:
                print_info("Using HeartMuLa engine (multilingual, supports tags)")
            engine = HeartMuLaStrategy()
        else:
            if verbose:
                print_info("Using MusicGen engine")
            engine = MusicGenStrategy()

        # Load model
        if verbose:
            print_info(f"Loading model: {model}")

        success = engine.load(config, device=device)
        if not success:
            print_error(f"Failed to load model: {model}")
            return

        if verbose:
            print_info("Generating music...")

        start_time = time.time()

        # Generate with appropriate parameters for each engine
        if isinstance(engine, HeartMuLaStrategy):
            result = engine.generate(
                prompt,
                duration=duration,
                tags=tags,
                temperature=temperature
            )
        else:
            result = engine.generate(
                prompt,
                duration=duration,
                temperature=temperature
            )

        elapsed = time.time() - start_time

        # Output
        if output:
            result.save(output)
            print_success(f"✓ Generated {result.duration:.1f}s of music")
            print_success(f"✓ Saved to {output}")
            print_info(f"Time: {elapsed:.1f}s")

            if play:
                _play_audio(output)
        else:
            # Save to temp file and play
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            result.save(temp_path)
            print_success(f"✓ Generated {result.duration:.1f}s of music")
            print_info(f"Time: {elapsed:.1f}s")
            print_info("Playing...")
            _play_audio(temp_path)
            os.unlink(temp_path)

        if verbose:
            info = engine.get_info()
            print_info(f"Engine: {info['engine']}")
            print_info(f"Device: {info['device']}")

    except Exception as e:
        print_error(f"Generation failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort()


@music.command("models")
@click.option("--verbose", is_flag=True, help="Show detailed information")
def models_cmd(verbose: bool):
    """
    List available music generation models.

    Shows memory requirements and supported durations.
    """
    print_header("Music Generation Models")

    try:
        from ...music import MusicGenStrategy, HeartMuLaStrategy

        print("\n" + "=" * 80)
        print("🎵 MUSICGEN (Meta) - General Purpose Music Generation")
        print("=" * 80)

        sizes = MusicGenStrategy.get_model_sizes()
        requirements = MusicGenStrategy.get_model_requirements()
        durations = MusicGenStrategy.get_supported_durations()

        for size in sizes:
            model_id = f"facebook/musicgen-{size}"
            req = requirements[size]
            print(f"\n  musicgen:{size}")
            print(f"    Model ID: {model_id}")
            print(f"    VRAM: {req['vram_gb']}GB | RAM: {req['ram_gb']}GB | Disk: {req['disk_gb']}GB")

        print(f"\n  Supported Durations: {', '.join(map(str, durations))} seconds")
        print("  Example: kin audio music generate 'calm piano' --model musicgen:small")

        print("\n" + "=" * 80)
        print("🎼 HEARTMULA (Open Source) - Multilingual, Tag-Based Control")
        print("=" * 80)

        sizes = HeartMuLaStrategy.get_model_sizes()
        requirements = HeartMuLaStrategy.get_model_requirements()
        durations = HeartMuLaStrategy.get_supported_durations()
        tags = HeartMuLaStrategy.get_available_tags()

        for size in sizes:
            req = requirements[size]
            print(f"\n  heartmula:{size}")
            print(f"    VRAM: {req['vram_gb']}GB | RAM: {req['ram_gb']}GB | Disk: {req['disk_gb']}GB")
            print(f"    {req['description']}")

        print(f"\n  Supported Durations: {', '.join(map(str, durations))} seconds")
        print(f"\n  Supported Languages: English, Chinese (中文), Japanese, Korean, Spanish")
        print(f"\n  Available Tags (examples):")
        # Print tags in columns
        tag_cols = 4
        for i in range(0, len(tags), tag_cols):
            tag_chunk = tags[i:i+tag_cols]
            print(f"    {', '.join(tag_chunk)}")

        print("\n  Examples:")
        print("    # Chinese lyrics")
        print("    kin audio music generate '在月光下弹钢琴' --model heartmula:3b")
        print("\n    # With style tags")
        print("    kin audio music generate 'happy music' --model heartmula:3b \\")
        print("      --tags 'piano,romantic,wedding' --duration 30")
        print("\n    # English lyrics")
        print("    kin audio music generate 'orchestral masterpiece' --model heartmula:3b")

        if verbose:
            print("\n" + "=" * 80)
            print("💡 Recommendation")
            print("=" * 80)
            print("  • HeartMuLa 3B: Recommended for most users (fast, good quality)")
            print("  • HeartMuLa 7B: Better quality, requires more VRAM")
            print("  • MusicGen Small: Quick generation on limited hardware")
            print("  • MusicGen Medium: Balance of quality and speed")

    except Exception as e:
        print_error(f"Failed to get models: {e}")


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
