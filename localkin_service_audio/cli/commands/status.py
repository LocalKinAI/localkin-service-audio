"""
Status command - Check system and model status.
"""
import click

from ..utils import print_header, print_success, print_error, print_warning, print_info


@click.command()
def status():
    """
    Check system and model status.

    Verifies library availability, model registry, cache, and configuration.

    Examples:

        kin audio status
    """
    print_header("System Status")

    # Check STT libraries
    print("\n📦 STT Libraries:")
    try:
        import whisper  # noqa: F401
        print_success("OpenAI Whisper: installed")
    except ImportError:
        print_warning("OpenAI Whisper: not installed")

    try:
        import faster_whisper  # noqa: F401
        print_success("faster-whisper: installed")
    except ImportError:
        print_warning("faster-whisper: not installed")

    try:
        import pywhispercpp  # noqa: F401
        print_success("whisper.cpp: installed")
    except ImportError:
        print_warning("whisper.cpp: not installed")

    # Check TTS libraries
    print("\n🔊 TTS Libraries:")
    try:
        import pyttsx3  # noqa: F401
        print_success("pyttsx3: installed")
    except ImportError:
        print_warning("pyttsx3: not installed")

    try:
        import kokoro  # noqa: F401
        print_success("kokoro: installed")
    except ImportError:
        print_warning("kokoro: not installed")

    # Check ML libraries
    print("\n🧠 ML Libraries:")
    try:
        import huggingface_hub  # noqa: F401
        print_success("Hugging Face Hub: installed")
    except ImportError:
        print_warning("Hugging Face Hub: not installed")

    try:
        import torch  # noqa: F401
        print_success(f"PyTorch: {torch.__version__}")
    except ImportError:
        print_warning("PyTorch: not installed")

    # Check model registry
    print("\n📋 Model Registry:")
    try:
        from ...core.config import model_registry
        all_models = model_registry.list_all()
        stt = model_registry.list_stt_models()
        tts = model_registry.list_tts_models()
        print_success(f"{len(all_models)} models registered ({len(stt)} STT, {len(tts)} TTS)")
    except Exception as e:
        print_error(f"Could not load model registry: {e}")

    # Check configuration
    print("\n⚙️  Configuration:")
    try:
        from ...core.config import settings
        print_success(f"Config dir: {settings.config_dir}")
        print_info(f"Cache dir: {settings.cache_dir}")
        print_info(f"Default STT: {settings.default_stt_model}")
        print_info(f"Default TTS: {settings.default_tts_model}")
    except Exception as e:
        print_error(f"Could not load settings: {e}")

    # Check cache
    print("\n💾 Cache:")
    try:
        from ...core.models import get_cache_info
        cache_info = get_cache_info()
        cached_models = cache_info.get("cached_models", [])
        if cached_models:
            total_mb = sum(m.get("size_mb", 0) for m in cached_models)
            print_success(f"{len(cached_models)} cached models ({total_mb:.1f} MB total)")
        else:
            print_info("No cached models")
    except Exception as e:
        print_warning(f"Could not read cache info: {e}")
