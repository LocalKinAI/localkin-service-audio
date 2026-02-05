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


def print_model_table(models: list, show_status: bool = True):
    """Print a formatted table of models."""
    if not models:
        print_warning("No models found.")
        return

    # Header
    if show_status:
        print(f"\n{'MODEL':<30} {'TYPE':<6} {'ENGINE':<15} {'STATUS':<12} {'DESCRIPTION'}")
        print("-" * 100)
    else:
        print(f"\n{'MODEL':<30} {'TYPE':<6} {'ENGINE':<15} {'DESCRIPTION'}")
        print("-" * 85)

    for model in models:
        name = getattr(model, 'name', str(model))
        model_type = getattr(model, 'type', 'N/A')
        if hasattr(model_type, 'value'):
            model_type = model_type.value
        engine = getattr(model, 'engine', 'N/A')
        description = getattr(model, 'description', 'No description')[:40]
        status = "📦 Available"

        if show_status:
            print(f"{name:<30} {model_type:<6} {engine:<15} {status:<12} {description}")
        else:
            print(f"{name:<30} {model_type:<6} {engine:<15} {description}")

    print(f"\n📊 Total: {len(models)} models")


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
