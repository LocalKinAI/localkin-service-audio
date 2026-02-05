"""
Config command - View and manage configuration settings.
"""
import click
import os
from pathlib import Path

from ..utils import print_header, print_success, print_info, print_error


def get_config_paths():
    """Get all configuration file paths."""
    return {
        "models_json": Path.home() / ".localkin-service-audio" / "models.json",
        "models_yaml": Path.home() / ".localkin-service-audio" / "models.yaml",
        "config_dir": Path.home() / ".localkin-service-audio",
        "cache_dir": Path.home() / ".cache" / "localkin-audio",
        "env_config": os.environ.get("LOCALKIN_MODEL_CONFIG"),
        "env_device": os.environ.get("LOCALKIN_DEVICE"),
    }


@click.command()
@click.option(
    "--path", "-p",
    is_flag=True,
    help="Show configuration file paths"
)
@click.option(
    "--models", "-m",
    is_flag=True,
    help="Show registered models summary"
)
@click.option(
    "--init",
    is_flag=True,
    help="Initialize config directory with sample config"
)
def config(path: bool, models: bool, init: bool):
    """
    View and manage configuration.

    Shows configuration paths, environment variables, and registered models.

    Examples:

        kin audio config

        kin audio config --path

        kin audio config --models

        kin audio config --init
    """
    print_header("Configuration")

    paths = get_config_paths()

    if init:
        _init_config(paths)
        return

    if path:
        _show_paths(paths)
        return

    if models:
        _show_models()
        return

    # Default: show overview
    _show_overview(paths)


def _show_overview(paths):
    """Show configuration overview."""
    print("\n📁 Config Directory:")
    config_dir = paths["config_dir"]
    if config_dir.exists():
        print_success(f"  {config_dir}")
    else:
        print_info(f"  {config_dir} (not created)")

    print("\n📄 Config Files:")
    for name, filepath in [
        ("models.json", paths["models_json"]),
        ("models.yaml", paths["models_yaml"]),
    ]:
        if filepath and filepath.exists():
            print_success(f"  {name}: {filepath}")
        else:
            print_info(f"  {name}: (not found)")

    print("\n🌍 Environment Variables:")
    env_config = paths["env_config"]
    env_device = paths["env_device"]
    if env_config:
        print_success(f"  LOCALKIN_MODEL_CONFIG={env_config}")
    else:
        print_info("  LOCALKIN_MODEL_CONFIG: (not set)")
    if env_device:
        print_success(f"  LOCALKIN_DEVICE={env_device}")
    else:
        print_info("  LOCALKIN_DEVICE: (not set)")

    print("\n💾 Cache Directory:")
    cache_dir = paths["cache_dir"]
    if cache_dir.exists():
        # Calculate cache size
        total_size = sum(
            f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print_success(f"  {cache_dir} ({size_mb:.1f} MB)")
    else:
        print_info(f"  {cache_dir} (not created)")

    # Show model counts
    print("\n📦 Registered Models:")
    try:
        from ...core.config import model_registry
        stt_count = len(model_registry.list_stt_models())
        tts_count = len(model_registry.list_tts_models())
        print_info(f"  STT models: {stt_count}")
        print_info(f"  TTS models: {tts_count}")
    except Exception as e:
        print_error(f"  Could not load registry: {e}")


def _show_paths(paths):
    """Show all configuration paths."""
    print("\n📁 Configuration Paths:\n")

    print("Config directory:")
    print(f"  {paths['config_dir']}")

    print("\nModel config files (checked in order):")
    print(f"  1. {paths['models_json']}")
    print(f"  2. {paths['models_yaml']}")
    if paths["env_config"]:
        print(f"  3. $LOCALKIN_MODEL_CONFIG={paths['env_config']}")

    print("\nCache directory:")
    print(f"  {paths['cache_dir']}")

    print("\nEnvironment variables:")
    print("  LOCALKIN_MODEL_CONFIG - Custom model config file path")
    print("  LOCALKIN_DEVICE - Default device (cpu, cuda, mps)")


def _show_models():
    """Show registered models summary."""
    try:
        from ...core.config import model_registry

        print("\n📦 STT Models:")
        for model in model_registry.list_stt_models():
            tags = ", ".join(model.tags[:3]) if model.tags else ""
            print(f"  {model.name:<30} [{model.engine}] {tags}")

        print("\n🔊 TTS Models:")
        for model in model_registry.list_tts_models():
            tags = ", ".join(model.tags[:3]) if model.tags else ""
            print(f"  {model.name:<30} [{model.engine}] {tags}")

    except Exception as e:
        print_error(f"Could not load models: {e}")


def _init_config(paths):
    """Initialize configuration directory with sample config."""
    config_dir = paths["config_dir"]

    # Create config directory
    config_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Created config directory: {config_dir}")

    # Create sample models.json
    sample_config = paths["models_json"]
    if not sample_config.exists():
        sample_content = '''{
  "models": {
    "my-custom-stt": {
      "type": "stt",
      "engine": "whisper",
      "model_size": "base",
      "languages": ["en", "zh"],
      "description": "My custom STT model",
      "tags": ["custom"]
    },
    "my-custom-tts": {
      "type": "tts",
      "engine": "kokoro",
      "languages": ["en"],
      "description": "My custom TTS model",
      "tags": ["custom"]
    }
  }
}
'''
        sample_config.write_text(sample_content)
        print_success(f"Created sample config: {sample_config}")
        print_info("Edit this file to add custom models.")
    else:
        print_info(f"Config already exists: {sample_config}")
