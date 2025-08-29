import argparse
import sys
import os
from pathlib import Path

# Import from the ollamaaudio package
try:
    from .config import get_models, find_model
    from .models import (
        list_local_models, pull_model, run_ollama_model, run_huggingface_model,
        get_cache_info, clear_cache
    )
    from .stt import transcribe_audio
    from .tts import synthesize_speech
except ImportError:
    # Fallback for direct execution
    from config import get_models, find_model
    from models import (
        list_local_models, pull_model, run_ollama_model,
        get_cache_info, clear_cache
    )
    from stt import transcribe_audio
    from tts import synthesize_speech

# Version information
__version__ = "0.1.0"

def print_header():
    """Print the ollamaaudio header."""
    print("🎵 OllamaAudio - Local STT & TTS Model Manager")
    print("=" * 50)

def print_success(message):
    """Print success message with green checkmark."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message with red X."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message with yellow warning."""
    print(f"⚠️  {message}")

def print_info(message):
    """Print info message with blue info."""
    print(f"ℹ️  {message}")

def handle_list(args):
    """Handles the 'list' command."""
    print_header()

    supported_models = get_models()
    local_ollama_models = list_local_models()

    if local_ollama_models is None:
        print_warning("Could not connect to Ollama to check for local models.")
        local_ollama_models = []

    if not supported_models:
        print_error("No models found in configuration.")
        return

    print(f"\n{'MODEL':<25} {'TYPE':<6} {'STATUS':<18} {'SOURCE':<15} {'DESCRIPTION'}")
    print("-" * 90)

    for model in supported_models:
        name = model.get('name', 'Unknown')
        model_type = model.get('type', 'N/A')
        source = model.get('source', 'unknown')
        description = model.get('description', 'No description available')

        # Determine status
        status = "N/A"
        if source == "ollama":
            status = "✅ Pulled" if any(m.startswith(name) for m in local_ollama_models) else "⬇️  Not Pulled"
        elif source in ["openai-whisper", "pyttsx3"]:
            status = "📦 Local Library"
        elif source == "huggingface":
            # Check if model is cached
            try:
                from .models import get_cache_info
                cache_info = get_cache_info()
                cached_models = [m["name"] for m in cache_info["cached_models"]]
                status = "✅ Pulled" if name in cached_models else "⬇️  Not Pulled"
            except Exception as e:
                print_warning(f"Cache check failed for {name}: {e}")
                status = "❓ Unknown"

        print(f"{name:<25} {model_type:<6} {status:<18} {source:<15} {description}")

    print(f"\n📊 Total models: {len(supported_models)}")

def handle_pull(args):
    """Handles the 'pull' command."""
    print_header()

    model_name = args.model_name
    print_info(f"Pulling model: {model_name}")

    model_info = find_model(model_name)
    if not model_info:
        print_error(f"Model '{model_name}' not found in configuration.")
        print_info("Use 'ollamaaudio list' to see available models.")
        return

    source = model_info.get("source")
    huggingface_repo = model_info.get("huggingface_repo")

    success = pull_model(model_name, source, huggingface_repo)
    if success:
        print_success(f"Successfully pulled model: {model_name}")
    else:
        print_error(f"Failed to pull model: {model_name}")

def handle_stt(args):
    """Handles the 'stt' command."""
    print_header()

    input_file = args.input_file
    model_size = args.model_size

    if not os.path.exists(input_file):
        print_error(f"Input file not found: {input_file}")
        return

    print_info(f"Transcribing audio file: {input_file}")
    print_info(f"Using Whisper model size: {model_size}")

    try:
        transcription = transcribe_audio(model_size, input_file)
        if transcription.startswith("Error:"):
            print_error(transcription)
        else:
            print_success("Transcription complete!")
            print("\n📝 Transcription:")
            print("-" * 40)
            print(transcription)
            print("-" * 40)
    except Exception as e:
        print_error(f"Transcription failed: {e}")

def handle_tts(args):
    """Handles the 'tts' command."""
    print_header()

    text = args.text
    output = args.output

    print_info("Synthesizing speech...")
    print_info(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")

    if output:
        print_info(f"Output file: {output}")

    try:
        success = synthesize_speech(text, output)
        if success:
            if output:
                print_success(f"Speech synthesized and saved to: {output}")
            else:
                print_success("Speech synthesized successfully!")
        else:
            print_error("Speech synthesis failed.")
    except Exception as e:
        print_error(f"Speech synthesis failed: {e}")

def handle_run(args):
    """Handles the 'run' command - runs a model server."""
    print_header()

    model_name = args.model_name
    port = getattr(args, 'port', 8000)

    print_info(f"Starting server for model: {model_name}")

    model_info = find_model(model_name)
    if not model_info:
        print_error(f"Model '{model_name}' not found in configuration.")
        return

    source = model_info.get('source')
    model_type = model_info.get('type')

    if source == 'ollama':
        success = run_ollama_model(model_name, port)
        if success:
            print_success(f"Model server started successfully for: {model_name}")
        else:
            print_error(f"Failed to start server for: {model_name}")
    elif source == 'huggingface':
        print_info(f"🚀 Starting API server for Hugging Face model: {model_name}")
        success = run_huggingface_model(model_name, port)
        if success:
            print_success(f"API server started successfully for: {model_name}")
        else:
            print_error(f"Failed to start API server for: {model_name}")
    elif source in ['openai-whisper', 'pyttsx3']:
        if model_type == 'stt':
            print_info("Local STT models don't require a server - use 'ollamaaudio stt' command instead.")
        elif model_type == 'tts':
            print_info("Local TTS models don't require a server - use 'ollamaaudio tts' command instead.")
    else:
        print_warning(f"Server mode not yet implemented for {source} models.")
        print_info("This feature will be available in future versions.")

def handle_status(args):
    """Handles the 'status' command."""
    print_header()

    print_info("Checking system status...")

    # Check Ollama connection
    try:
        local_models = list_local_models()
        if local_models is not None:
            print_success(f"Ollama connection: OK ({len(local_models)} models available)")
        else:
            print_error("Ollama connection: Failed")
    except Exception as e:
        print_error(f"Ollama connection: Failed - {e}")

    # Check local libraries
    try:
        import whisper
        print_success("OpenAI Whisper: Available")
    except ImportError:
        print_warning("OpenAI Whisper: Not installed")

    try:
        import pyttsx3
        print_success("pyttsx3: Available")
    except ImportError:
        print_warning("pyttsx3: Not installed")

    try:
        import huggingface_hub
        print_success("Hugging Face Hub: Available")
    except ImportError:
        print_warning("Hugging Face Hub: Not installed")

    # Check configuration
    models = get_models()
    if models:
        print_success(f"Configuration: OK ({len(models)} models configured)")
    else:
        print_error("Configuration: No models found")

    # Check cache
    try:
        from .models import get_cache_info
        cache_info = get_cache_info()
        cached_count = len(cache_info["cached_models"])
        if cached_count > 0:
            total_size = sum(model["size_mb"] for model in cache_info["cached_models"])
            print_success(f"Cache: OK ({cached_count} models, {total_size:.1f}MB)")
        else:
            print_info("Cache: Empty")
    except Exception as e:
        print_warning(f"Cache check failed: {e}")

def handle_cache(args):
    """Handles the 'cache' command."""
    print_header()

    subcommand = getattr(args, 'subcommand', 'info')

    if subcommand == 'info':
        cache_info = get_cache_info()
        print_info("Cache Information:")
        print(f"📁 Cache Directory: {cache_info['cache_dir']}")
        print(f"🤗 HF Cache Directory: {cache_info['huggingface_cache']}")

        if cache_info["cached_models"]:
            print(f"\n📦 Cached Models ({len(cache_info['cached_models'])}):")
            for model in cache_info["cached_models"]:
                print(f"  • {model['name']} ({model['size_mb']}MB) - {model['path']}")
        else:
            print("\n📦 No cached models")

    elif subcommand == 'clear':
        model_name = getattr(args, 'model_name', None)
        if model_name:
            success = clear_cache(model_name)
        else:
            # Confirm before clearing all
            print_warning("This will clear ALL cached models!")
            response = input("Are you sure? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                success = clear_cache()
            else:
                print_info("Operation cancelled.")
                return

        if success:
            print_success("Cache cleared successfully")
        else:
            print_error("Failed to clear cache")

def handle_version(args):
    """Handles the 'version' command."""
    print_header()
    print(f"🎵 OllamaAudio version {__version__}")
    print(f"📍 Location: {Path(__file__).parent.absolute()}")

def main():
    parser = argparse.ArgumentParser(
        description="🎵 OllamaAudio - Local STT & TTS Model Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ollamaaudio list                    # List all available models
  ollamaaudio pull whisper-large-v3   # Pull an Ollama model
  ollamaaudio pull whisper-tiny-hf    # Pull from Hugging Face
  ollamaaudio run llama3.2:3b         # Run Ollama model server
  ollamaaudio stt audio.wav           # Transcribe audio file
  ollamaaudio tts "Hello world"       # Generate speech
  ollamaaudio cache info              # Check cache status
  ollamaaudio status                  # Check system status
        """
    )

    parser.add_argument('-v', '--version', action='version', version=f'OllamaAudio {__version__}')

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    parser_list = subparsers.add_parser("list", help="List all supported and locally available models.")
    parser_list.add_argument("--verbose", "-v", action="store_true", help="Show detailed model information.")
    parser_list.set_defaults(func=handle_list)

    # Pull command
    parser_pull = subparsers.add_parser("pull", help="Pull a model from Ollama or Hugging Face.")
    parser_pull.add_argument("model_name", help="The name of the model to pull.")
    parser_pull.set_defaults(func=handle_pull)

    # Run command
    parser_run = subparsers.add_parser("run", help="Run a model server.")
    parser_run.add_argument("model_name", help="The name of the model to run.")
    parser_run.add_argument("--port", "-p", type=int, default=8000, help="Port for the model server.")
    parser_run.set_defaults(func=handle_run)

    # STT command
    parser_stt = subparsers.add_parser("stt", help="Perform Speech-to-Text using local models.")
    parser_stt.add_argument("input_file", help="Path to the audio file to transcribe.")
    parser_stt.add_argument("--model", "-m", default="whisper", help="The STT model to use.")
    parser_stt.add_argument("--model_size", default="base", help="The size of the Whisper model to use.")
    parser_stt.set_defaults(func=handle_stt)

    # TTS command
    parser_tts = subparsers.add_parser("tts", help="Perform Text-to-Speech using local models.")
    parser_tts.add_argument("text", help="The text to synthesize.")
    parser_tts.add_argument("--output", "-o", help="Path to save the output audio file.")
    parser_tts.add_argument("--model", "-m", default="native", help="The TTS model to use.")
    parser_tts.set_defaults(func=handle_tts)

    # Status command
    parser_status = subparsers.add_parser("status", help="Check system and model status.")
    parser_status.set_defaults(func=handle_status)

    # Cache command
    parser_cache = subparsers.add_parser("cache", help="Manage model cache.")
    cache_subparsers = parser_cache.add_subparsers(dest="subcommand", help="Cache subcommands")

    # Cache info
    parser_cache_info = cache_subparsers.add_parser("info", help="Show cache information.")
    parser_cache_info.set_defaults(func=handle_cache, subcommand="info")

    # Cache clear
    parser_cache_clear = cache_subparsers.add_parser("clear", help="Clear cache.")
    parser_cache_clear.add_argument("model_name", nargs="?", help="Model name to clear (optional - clears all if not specified).")
    parser_cache_clear.set_defaults(func=handle_cache, subcommand="clear")

    # Default cache command
    parser_cache.set_defaults(func=handle_cache, subcommand="info")

    # Version command
    parser_version = subparsers.add_parser("version", help="Show version information.")
    parser_version.set_defaults(func=handle_version)

    # If no command provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
