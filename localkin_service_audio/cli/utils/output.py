"""
Output formatting utilities for CLI.
"""
from typing import Optional


def print_header(title: Optional[str] = None):
    """Print the localkin-service-audio header."""
    if title:
        print(f"🎵 LocalKin Audio - {title}")
    else:
        print("🎵 LocalKin Audio - Voice AI Platform")
    print("=" * 60)


def print_success(message: str):
    """Print success message with green checkmark."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message with red X."""
    print(f"❌ {message}")


def print_warning(message: str):
    """Print warning message with yellow warning."""
    print(f"⚠️  {message}")


def print_info(message: str):
    """Print info message with blue info."""
    print(f"ℹ️  {message}")


def _check_engine_installed(engine: str) -> bool:
    """Check if an engine's required library is installed."""
    import importlib
    engine_imports = {
        "whisper": "whisper",
        "openai-whisper": "whisper",
        "faster-whisper": "faster_whisper",
        "whisper-cpp": "pywhispercpp",
        "sensevoice": "funasr",
        "funasr": "funasr",
        "paraformer": "funasr",
        "moonshine": "moonshine_onnx",
        "native": "pyttsx3",
        "pyttsx3": "pyttsx3",
        "kokoro": "kokoro",
        "chattts": "ChatTTS",
        "f5-tts": "f5_tts",
        "f5": "f5_tts",
        "mlx-audio": "mlx_audio",
    }
    if engine == "isolated":
        # Builds its own environment on first use; only uv is needed.
        from ...core.audio_processing.isolated import find_uv
        return find_uv() is not None
    pkg = engine_imports.get(engine)
    if not pkg:
        return False
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


# Engines with NO strategy implementation (future work). Empty since every
# registry entry now has one — test_every_registry_model_has_an_implementation
# keeps it that way.
_PLANNED_ENGINES: set = set()

# Cache so we only probe imports once per session
_engine_status_cache: dict = {}


def _get_engine_status(engine: str) -> str:
    """Return status string for an engine: ready / install needed / planned."""
    if engine in _engine_status_cache:
        return _engine_status_cache[engine]

    if engine in _PLANNED_ENGINES:
        status = "planned"
    elif _check_engine_installed(engine):
        status = "ready"
    else:
        status = "install"
    _engine_status_cache[engine] = status
    return status


_STATUS_LABELS = {
    "ready": "✅ Ready",
    "install": "📦 Not installed",
    "planned": "🔮 Planned",
}


def print_model_table(models: list, show_status: bool = True):
    """Print a formatted table of models."""
    if not models:
        print_warning("No models found.")
        return

    # Header
    print(f"\n{'MODEL':<30} {'TYPE':<6} {'ENGINE':<15} {'STATUS':<18} {'DESCRIPTION'}")
    print("-" * 105)

    counts = {"ready": 0, "install": 0, "planned": 0}

    from ...core.config.backends import resolve_backend

    for model in models:
        # Multi-backend models show the backend this machine would use.
        model = resolve_backend(model) if hasattr(model, "backends") else model
        name = getattr(model, 'name', str(model))
        model_type = getattr(model, 'type', 'N/A')
        if hasattr(model_type, 'value'):
            model_type = model_type.value
        engine = getattr(model, 'engine', 'N/A')
        description = getattr(model, 'description', 'No description')[:40]
        status = _get_engine_status(engine)
        counts[status] = counts.get(status, 0) + 1
        label = _STATUS_LABELS.get(status, status)

        print(f"{name:<30} {model_type:<6} {engine:<15} {label:<18} {description}")

    parts = []
    if counts["ready"]:
        parts.append(f"{counts['ready']} ready")
    if counts["install"]:
        parts.append(f"{counts['install']} need install")
    if counts["planned"]:
        parts.append(f"{counts['planned']} planned")
    print(f"\n📊 Total: {len(models)} models ({', '.join(parts)})")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_size(bytes_size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} PB"
