"""
Config command - View and manage configuration settings.
"""
import click
import os
from pathlib import Path

from ..utils import print_header, print_success, print_info, print_error, print_warning


# Valid keys for `config set`
SETTABLE_KEYS = {
    "cache_dir": "Cache directory path",
    "config_dir": "Config directory path",
    "models_dir": "Models directory path",
    "default_stt_model": "Default STT model (e.g. whisper-cpp:base)",
    "default_tts_model": "Default TTS model (e.g. kokoro)",
    "api_host": "API server host (e.g. 127.0.0.1)",
    "api_port": "API server port (e.g. 8000)",
    "default_device": "Default device (auto, cpu, cuda, mps)",
}


def _get_settings():
    """Get the settings singleton."""
    from ...core.config import settings
    return settings


@click.group(invoke_without_command=True)
@click.option("--path", "-p", is_flag=True, help="Show configuration file paths")
@click.option("--models", "-m", is_flag=True, help="Show registered models summary")
@click.option("--init", is_flag=True, help="Initialize config directory with sample config")
@click.pass_context
def config(ctx, path: bool, models: bool, init: bool):
    """
    View and manage configuration.

    Shows configuration paths, environment variables, and registered models.

    Examples:

        kin audio config

        kin audio config --path

        kin audio config --models

        kin audio config --init

        kin audio config set cache_dir /tmp/my-cache

        kin audio config set default_stt_model faster-whisper:large-v3
    """
    if ctx.invoked_subcommand is not None:
        return

    print_header("Configuration")

    s = _get_settings()

    if init:
        _init_config(s)
        return

    if path:
        _show_paths(s)
        return

    if models:
        _show_models()
        return

    # Default: show overview
    _show_overview(s)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """
    Set a configuration value.

    Saves to ~/.localkin-service-audio/config.json.

    Available keys:

        cache_dir, config_dir, models_dir,
        default_stt_model, default_tts_model,
        api_host, api_port, default_device

    Examples:

        kin audio config set cache_dir /tmp/my-cache

        kin audio config set default_stt_model faster-whisper:large-v3

        kin audio config set api_port 9000

        kin audio config set default_device cuda
    """
    if key not in SETTABLE_KEYS:
        print_error(f"Unknown key: {key}")
        print_info("Available keys:")
        for k, desc in SETTABLE_KEYS.items():
            print(f"  {k:<25} {desc}")
        raise click.Abort()

    s = _get_settings()

    # Convert value to correct type
    if key.endswith("_dir"):
        setattr(s, key, Path(value))
    elif key == "api_port":
        try:
            setattr(s, key, int(value))
        except ValueError:
            print_error(f"api_port must be an integer, got: {value}")
            raise click.Abort()
    else:
        setattr(s, key, value)

    s.save()
    print_success(f"{key} = {value}")
    print_info(f"Saved to {s.config_dir / 'config.json'}")


def _show_overview(s):
    """Show configuration overview."""
    print("\n📁 Directories:")
    print_success(f"  Config:  {s.config_dir}")
    print_success(f"  Cache:   {s.cache_dir}")
    print_success(f"  Models:  {s.models_dir}")

    # Check for user config files
    print("\n📄 Config Files:")
    config_json = s.config_dir / "config.json"
    models_json = s.config_dir / "models.json"
    models_yaml = s.config_dir / "models.yaml"
    for name, filepath in [
        ("config.json", config_json),
        ("models.json", models_json),
        ("models.yaml", models_yaml),
    ]:
        if filepath.exists():
            print_success(f"  {name}: {filepath}")
        else:
            print_info(f"  {name}: (not found)")

    print("\n⚙️  Settings:")
    print_info(f"  Default STT: {s.default_stt_model}")
    print_info(f"  Default TTS: {s.default_tts_model}")
    print_info(f"  API host:    {s.api_host}:{s.api_port}")
    print_info(f"  Device:      {s.default_device}")

    print("\n🌍 Environment Overrides:")
    env_vars = {
        "LOCALKIN_CACHE_DIR": "cache_dir",
        "LOCALKIN_CONFIG_DIR": "config_dir",
        "LOCALKIN_MODELS_DIR": "models_dir",
        "LOCALKIN_DEFAULT_STT": "default_stt_model",
        "LOCALKIN_DEFAULT_TTS": "default_tts_model",
        "LOCALKIN_API_HOST": "api_host",
        "LOCALKIN_API_PORT": "api_port",
        "LOCALKIN_DEVICE": "default_device",
    }
    any_set = False
    for env_var, attr in env_vars.items():
        val = os.environ.get(env_var)
        if val:
            print_success(f"  {env_var}={val}")
            any_set = True
    if not any_set:
        print_info("  (none set)")

    # Cache size
    print("\n💾 Cache:")
    if s.cache_dir.exists():
        try:
            total_size = sum(
                f.stat().st_size for f in s.cache_dir.rglob("*") if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            print_success(f"  {size_mb:.1f} MB used")
        except Exception:
            print_info("  (could not calculate size)")
    else:
        print_info("  (empty)")

    # Model counts
    print("\n📦 Registered Models:")
    try:
        from ...core.config import model_registry
        stt_count = len(model_registry.list_stt_models())
        tts_count = len(model_registry.list_tts_models())
        print_info(f"  STT: {stt_count}  TTS: {tts_count}")
    except Exception as e:
        print_error(f"  Could not load registry: {e}")


def _show_paths(s):
    """Show all configuration paths."""
    print("\n📁 All Paths:\n")

    print(f"  Config dir:  {s.config_dir}")
    print(f"  Cache dir:   {s.cache_dir}")
    print(f"  Models dir:  {s.models_dir}")
    print(f"  config.json: {s.config_dir / 'config.json'}")
    print(f"  models.json: {s.config_dir / 'models.json'}")
    print(f"  models.yaml: {s.config_dir / 'models.yaml'}")

    print("\n🌍 Environment variable overrides:")
    print("  LOCALKIN_CACHE_DIR      -> cache_dir")
    print("  LOCALKIN_CONFIG_DIR     -> config_dir")
    print("  LOCALKIN_MODELS_DIR     -> models_dir")
    print("  LOCALKIN_DEFAULT_STT    -> default_stt_model")
    print("  LOCALKIN_DEFAULT_TTS    -> default_tts_model")
    print("  LOCALKIN_API_HOST       -> api_host")
    print("  LOCALKIN_API_PORT       -> api_port")
    print("  LOCALKIN_DEVICE         -> default_device")


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


def _init_config(s):
    """Initialize configuration directory with sample config."""
    s.config_dir.mkdir(parents=True, exist_ok=True)
    print_success(f"Config directory: {s.config_dir}")

    # Create sample models.json if not exists
    sample_config = s.config_dir / "models.json"
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

    # Save default config.json
    config_json = s.config_dir / "config.json"
    if not config_json.exists():
        s.save()
        print_success(f"Created settings: {config_json}")
    else:
        print_info(f"Settings already exist: {config_json}")
