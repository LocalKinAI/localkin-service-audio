"""
Models command - Model management operations.
"""
import click
from typing import Optional

from ..utils import print_success, print_error, print_info, print_header, print_model_table


@click.command()
@click.option(
    "--type", "-t",
    type=click.Choice(["all", "stt", "tts"]),
    default="all",
    help="Filter by model type."
)
@click.option(
    "--engine", "-e",
    default=None,
    help="Filter by engine (e.g., whisper, kokoro, cosyvoice)."
)
@click.option(
    "--language", "-l",
    default=None,
    help="Filter by supported language (e.g., en, zh, ja)."
)
@click.option(
    "--tag",
    default=None,
    help="Filter by tag (e.g., fast, chinese, voice-cloning)."
)
@click.option(
    "--search", "-s",
    default=None,
    help="Search models by name or description."
)
def models(
    type: str,
    engine: Optional[str],
    language: Optional[str],
    tag: Optional[str],
    search: Optional[str]
):
    """
    List available models.

    Examples:

        kin audio models

        kin audio models --type stt

        kin audio models --language zh

        kin audio models --tag voice-cloning

        kin audio models --search whisper
    """
    print_header("Available Models")

    try:
        from ...core.config import model_registry
        from ...core.types import ModelType

        # Get models
        if type == "stt":
            all_models = model_registry.list_stt_models()
        elif type == "tts":
            all_models = model_registry.list_tts_models()
        else:
            all_models = model_registry.list_all()

        # Apply filters
        if engine:
            all_models = [m for m in all_models if m.engine == engine]

        if language:
            all_models = [m for m in all_models if language in m.languages]

        if tag:
            all_models = [m for m in all_models if tag in m.tags]

        if search:
            search_lower = search.lower()
            all_models = [
                m for m in all_models
                if search_lower in m.name.lower() or
                   (m.description and search_lower in m.description.lower())
            ]

        # Sort: STT first, then by name
        stt_models = sorted([m for m in all_models if m.type == ModelType.STT], key=lambda x: x.name)
        tts_models = sorted([m for m in all_models if m.type == ModelType.TTS], key=lambda x: x.name)

        if stt_models:
            print("\n🎤 Speech-to-Text (STT) Models:")
            print_model_table(stt_models)

        if tts_models:
            print("\n🔊 Text-to-Speech (TTS) Models:")
            print_model_table(tts_models)

        if not stt_models and not tts_models:
            print_info("No models found matching your criteria.")

    except Exception as e:
        print_error(f"Failed to list models: {e}")
        raise click.Abort()


@click.command()
@click.argument("model_name")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force re-download even if model exists."
)
def pull(model_name: str, force: bool):
    """
    Download a model.

    Examples:

        kin audio pull whisper:large-v3

        kin audio pull cosyvoice:300m --force
    """
    print_header("Pull Model")
    print_info(f"Pulling model: {model_name}")

    try:
        from ...core.config import model_registry
        from ..utils.progress import progress

        # Check if model exists in registry
        model_config = model_registry.get(model_name)
        if not model_config:
            print_error(f"Model '{model_name}' not found in registry.")
            print_info("Use 'kin audio models' to see available models.")
            return

        # For now, just show info about the model
        # Actual download depends on the engine
        print_info(f"Model: {model_config.name}")
        print_info(f"Engine: {model_config.engine}")
        print_info(f"Type: {model_config.type.value}")

        if model_config.repo_id:
            print_info(f"Repository: {model_config.repo_id}")

        # Try to load the model (which triggers download)
        if model_config.engine == "heartmula":
            # Music generation models use their own strategy
            from ...music import HeartMuLaStrategy

            engine = HeartMuLaStrategy()
            with progress.spinner(f"Setting up {model_name}"):
                success = engine.load(model_config, device="auto")
            if success:
                engine.unload()
        else:
            from ...core import get_audio_engine
            engine = get_audio_engine()

            with progress.spinner(f"Downloading {model_name}"):
                if model_config.type.value == "stt":
                    success = engine.load_stt(model_name)
                else:
                    success = engine.load_tts(model_name)

        if success:
            print_success(f"Successfully pulled {model_name}")
        else:
            print_error(f"Failed to pull {model_name}")

    except Exception as e:
        print_error(f"Failed to pull model: {e}")
        raise click.Abort()


@click.command()
@click.argument("model_name")
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Force remove without confirmation."
)
def rm(model_name: str, force: bool):
    """
    Remove a downloaded model.

    Examples:

        kin audio rm whisper:large-v3

        kin audio rm cosyvoice:300m --force
    """
    print_header("Remove Model")

    if not force:
        if not click.confirm(f"Are you sure you want to remove '{model_name}'?"):
            print_info("Cancelled.")
            return

    try:
        from ...core.config import settings
        import shutil

        # Get model path
        model_path = settings.get_model_path(model_name)

        if model_path.exists():
            shutil.rmtree(model_path)
            print_success(f"Removed {model_name}")
        else:
            print_info(f"Model {model_name} is not downloaded locally.")

    except Exception as e:
        print_error(f"Failed to remove model: {e}")
        raise click.Abort()
