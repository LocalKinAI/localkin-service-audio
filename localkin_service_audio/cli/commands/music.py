"""
Music generation command - AI music generation operations.
"""
import click
import json
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

        kin audio music generate "lofi, rainy night" --model minimax-music3 --lyrics @song.txt
    """
    pass


@music.command("generate")
@click.argument("prompt")
@click.option(
    "--model", "-m",
    default="musicgen:small",
    help="Model to use: musicgen:small/medium/large, heartmula:3b/7b, or a ComfyUI model "
         "(minimax-music3, ace-step:1.5, yue2, stable-audio3, comfyui:<blueprint name>)"
)
@click.option(
    "--tags",
    default=None,
    help="Music style tags (for HeartMuLa: piano,happy,wedding,etc.)"
)
@click.option(
    "--duration", "-d",
    type=int,
    default=None,
    help="Duration in seconds (MusicGen 5-30, default 10; HeartMuLa up to 240; "
         "ComfyUI models: the blueprint's default)"
)
@click.option(
    "--lyrics", "-l",
    default=None,
    help="Lyrics, or @path to read them from a file (MiniMax Music 3, YuE2, ACE-Step)."
)
@click.option("--seed", type=int, default=None, help="Seed for reproducible results.")
@click.option(
    "--comfyui-url",
    default=None,
    help="ComfyUI address for ComfyUI models (default $LOCALKIN_COMFYUI_URL or http://localhost:8188)."
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "mlx", "comfyui"]),
    default="auto",
    help="For models with both: mlx (fast on Apple Silicon) or comfyui. auto picks mlx when available."
)
@click.option(
    "--param", "params",
    multiple=True,
    help="Set any blueprint input directly, e.g. --param cfg_scale=2.0 (ComfyUI models)."
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
    duration: Optional[int],
    lyrics: Optional[str],
    seed: Optional[int],
    comfyui_url: Optional[str],
    backend: str,
    params: tuple,
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

        # Through ComfyUI (weights installed there)
        kin audio music generate "欢快的流行，女声" --model minimax-music3 --lyrics @lyrics.txt -o song.flac
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
        from ...music.comfyui_strategy import ComfyUIMusicStrategy, is_comfyui_model
        from ...music.mlx_music_strategy import MLXMusicStrategy, mlx_music_available

        if lyrics and lyrics.startswith("@"):
            with open(os.path.expanduser(lyrics[1:]), encoding="utf-8") as f:
                lyrics = f.read()

        # Create model config
        config = ModelConfig(
            name=model,
            type=ModelType.TTS,  # Use TTS as fallback for music generation
            engine="music"
        )

        # Select engine based on model name
        use_mlx = backend == "mlx" or (backend == "auto" and mlx_music_available(model))
        if use_mlx:
            if verbose:
                print_info("Using mlx-audio")
            engine = MLXMusicStrategy()
        elif is_comfyui_model(model):
            if verbose:
                print_info(f"Using ComfyUI at {comfyui_url or 'default address'}")
            engine = ComfyUIMusicStrategy(comfyui_url)
        elif model.startswith("heartmula") or model.startswith("heart"):
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
            reason = getattr(engine, "load_error", None)
            print_error(f"Failed to load model: {model}" + (f" — {reason}" if reason else ""))
            return

        if verbose:
            print_info("Generating music...")

        start_time = time.time()

        # Generate with appropriate parameters for each engine
        if isinstance(engine, MLXMusicStrategy):
            print_info("Generating with mlx-audio; a full song takes a while...")
            result = engine.generate(prompt, duration=duration, lyrics=lyrics, seed=seed, tags=tags)
        elif isinstance(engine, ComfyUIMusicStrategy):
            extra = {}
            for item in params:
                key, _, value = item.partition("=")
                try:
                    extra[key] = json.loads(value)
                except ValueError:
                    extra[key] = value
            print_info(f"Queued on ComfyUI ({engine.url}); a full song can take a few minutes...")
            result = engine.generate(
                prompt, duration=duration, lyrics=lyrics, seed=seed, tags=tags, params=extra
            )
        elif isinstance(engine, HeartMuLaStrategy):
            result = engine.generate(
                prompt,
                duration=duration or 10,
                tags=tags,
                temperature=temperature
            )
        else:
            result = engine.generate(
                prompt,
                duration=duration or 10,
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
    import importlib.util
    from types import SimpleNamespace

    from ..utils.output import print_model_table
    from ...music import MusicGenStrategy, HeartMuLaStrategy
    from ...music.comfyui_strategy import MODELS as COMFY_MODELS, comfyui_url, list_audio_blueprints
    from ...music.mlx_music_strategy import mlx_music_available

    def row(name, engine, status, description):
        return SimpleNamespace(name=name, type="music", engine=engine, status=status, description=description)

    has = lambda pkg: importlib.util.find_spec(pkg) is not None
    rows = []
    for size, req in MusicGenStrategy.get_model_requirements().items():
        rows.append(row(f"musicgen:{size}", "transformers", "ready" if has("transformers") else "install",
                        f"MusicGen {size} - instrumental, 5-30s, {req['vram_gb']}GB"))
    for size, req in HeartMuLaStrategy.get_model_requirements().items():
        rows.append(row(f"heartmula:{size}", "heartmula", "ready" if has("heartlib") else "install",
                        f"HeartMuLa {size} - songs with zh/en lyrics, {req['vram_gb']}GB"))

    try:
        blueprints = {m["model"]: m for m in list_audio_blueprints()}
    except Exception:
        blueprints = None

    mlx_ok = mlx_music_available("minimax-music3")
    comfy_minimax = (blueprints or {}).get("minimax-music3")
    status = "ready" if mlx_ok or (comfy_minimax and comfy_minimax["ready"]) else \
        ("offline" if blueprints is None else "weights")
    rows.append(row("minimax-music3", "mlx-audio" if mlx_ok else "comfyui", status,
                    "MiniMax Music 3 - full songs with lyrics"))

    if blueprints is None:
        for name in COMFY_MODELS:
            if name != "minimax-music3":
                rows.append(row(name, "comfyui", "offline", COMFY_MODELS[name]))
    else:
        for name, m in blueprints.items():
            if name == "minimax-music3":
                continue
            description = m["blueprint"] if m["ready"] else "needs " + ", ".join(m["missing"])
            rows.append(row(name, "comfyui", "ready" if m["ready"] else "weights", description))

    print("\n🎵 Music Generation Models:")
    print_model_table(rows, status_of=lambda r: r.status)
    print(f"\nComfyUI: {comfyui_url()}" + ("  (unreachable — set LOCALKIN_COMFYUI_URL)" if blueprints is None else ""))

    if verbose:
        print("\nExamples:")
        print("  kin audio music generate 'calm piano melody' --model musicgen:small")
        print("  kin audio music generate '在月光下弹钢琴' --model heartmula:3b --tags 'piano,romantic'")
        print("  kin audio music generate '中文流行，女声' --model minimax-music3 --lyrics @song.txt --duration 60")
        print("  kin audio music generate 'rain ambience' --model stable-audio3")
        print("\nHeartMuLa tags: " + ", ".join(HeartMuLaStrategy.get_available_tags()))
        if blueprints:
            print("\nComfyUI blueprint inputs (set with --param name=value):")
            for name, m in blueprints.items():
                print(f"  {name:<22} {', '.join(m['inputs'])}")


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
