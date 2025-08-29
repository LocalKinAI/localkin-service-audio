import argparse
import sys
import os
from pathlib import Path

# Import from the ollamaaudio package
try:
    from .config import get_models, find_model, save_models_config, get_config_metadata
    from .models import (
        list_local_models, pull_model, run_ollama_model, run_huggingface_model,
        get_cache_info, clear_cache
    )
    from .stt import transcribe_audio
    from .tts import synthesize_speech
    from .model_templates import (
        get_model_template, list_available_templates, create_model_from_template,
        validate_model_for_huggingface, suggest_similar_models, get_popular_models
    )
except ImportError:
    # Fallback for direct execution
    from config import get_models, find_model, save_models_config, get_config_metadata
    from models import (
        list_local_models, pull_model, run_ollama_model,
        get_cache_info, clear_cache
    )
    from stt import transcribe_audio
    from tts import synthesize_speech
    from model_templates import (
        get_model_template, list_available_templates, create_model_from_template,
        validate_model_for_huggingface, suggest_similar_models, get_popular_models
    )

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

    # Show detailed model information
    print_info(f"🎵 Transcribing audio file: {input_file}")
    print_info(f"🤖 Using Whisper model: {model_size}")

    # Get model details
    model_details = {
        "tiny": {"size": "39MB", "speed": "32x", "quality": "Basic"},
        "base": {"size": "74MB", "speed": "16x", "quality": "Good"},
        "small": {"size": "244MB", "speed": "8x", "quality": "High"},
        "medium": {"size": "769MB", "speed": "4x", "quality": "Very High"},
        "large": {"size": "1550MB", "speed": "1x", "quality": "Excellent"}
    }

    if model_size in model_details:
        details = model_details[model_size]
        print_info(f"📊 Model details: {details['size']} | {details['speed']} speed | {details['quality']} quality")
    else:
        print_warning(f"Unknown model size: {model_size}")

    print_info("🔄 Processing audio...")

    try:
        transcription = transcribe_audio(model_size, input_file)
        if transcription.startswith("Error:"):
            print_error(transcription)
        else:
            print_success("✅ Transcription complete!")
            print("\n📝 Transcription Result:")
            print("=" * 60)
            print(transcription)
            print("=" * 60)

            # Show statistics
            word_count = len(transcription.split())
            char_count = len(transcription)
            print_info(f"📊 Statistics: {word_count} words, {char_count} characters")

    except Exception as e:
        print_error(f"❌ Transcription failed: {e}")

def handle_tts(args):
    """Handles the 'tts' command."""
    print_header()

    text = args.text
    output = args.output

    # Show detailed model information
    print_info("🔊 Synthesizing speech...")
    print_info("🤖 Using TTS engine: pyttsx3 (native OS TTS)")

    # Show text preview
    text_preview = text[:100] + "..." if len(text) > 100 else text
    print_info(f"📝 Text: {text_preview}")

    # Show text statistics
    word_count = len(text.split())
    char_count = len(text)
    print_info(f"📊 Text statistics: {word_count} words, {char_count} characters")

    if output:
        print_info(f"💾 Output file: {output}")
        # Get file extension info
        if output.lower().endswith('.wav'):
            print_info("🎵 Output format: WAV (uncompressed)")
        elif output.lower().endswith('.mp3'):
            print_info("🎵 Output format: MP3 (compressed)")
        else:
            print_info("🎵 Output format: System default")
    else:
        print_info("🔊 Output: Playing through system speakers")

    print_info("🔄 Processing text...")

    try:
        success = synthesize_speech(text, output)
        if success:
            if output:
                print_success(f"✅ Speech synthesized and saved to: {output}")
                # Show file size if possible
                try:
                    file_size = os.path.getsize(output) / (1024 * 1024)  # MB
                    print_info(".2f")
                except:
                    pass
            else:
                print_success("✅ Speech synthesized and played successfully!")
        else:
            print_error("❌ Speech synthesis failed.")
    except Exception as e:
        print_error(f"❌ Speech synthesis failed: {e}")

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

def handle_ps(args):
    """Handles the 'ps' command to show running processes."""
    print_header()

    import subprocess
    import socket

    print_info("Checking for running OllamaAudio processes...")

    running_servers = []

    # Check common ports for OllamaAudio servers
    common_ports = [8000, 8001, 8002, 8003, 8004, 8005]

    for port in common_ports:
        try:
            # Check if port is in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()

            if result == 0:
                # Port is open, check if it's an OllamaAudio server
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/", timeout=2)
                    if response.status_code == 200 and "OllamaAudio" in response.text:
                        # Try to get model info
                        try:
                            model_response = requests.get(f"http://localhost:{port}/models", timeout=2)
                            if model_response.status_code == 200:
                                model_data = model_response.json()
                                model_name = model_data.get("model_name", "Unknown")
                                model_type = model_data.get("model_type", "Unknown")
                            else:
                                model_name = "Unknown"
                                model_type = "Unknown"
                        except:
                            model_name = "Unknown"
                            model_type = "Unknown"

                        running_servers.append({
                            "port": port,
                            "model": model_name,
                            "type": model_type,
                            "url": f"http://localhost:{port}"
                        })
                except:
                    # Port is open but not an OllamaAudio server
                    continue
        except:
            continue

    if running_servers:
        print_success(f"Found {len(running_servers)} running OllamaAudio server(s):")
        print("\n" + "=" * 80)
        print(f"{'PORT':<8} {'MODEL':<25} {'TYPE':<8} {'URL':<25} {'STATUS'}")
        print("=" * 80)

        for server in running_servers:
            try:
                # Quick health check
                import requests
                health_response = requests.get(f"http://localhost:{server['port']}/health", timeout=2)
                if health_response.status_code == 200:
                    status = "🟢 Running"
                else:
                    status = "🟡 Issues"
            except:
                status = "🔴 Offline"

            print(f"{server['port']:<8} {server['model']:<25} {server['type']:<8} {server['url']:<25} {status}")

        print("=" * 80)
        print(f"\n💡 Tip: Access interactive API docs at http://localhost:<PORT>/docs")
    else:
        print_info("No running OllamaAudio servers found.")
        print("💡 Start a server with: ollamaaudio run <model_name> --port <port>")

def handle_add_model(args):
    """Handles the 'add-model' command."""
    print_header()

    # Get current models
    current_models = get_models()
    current_names = [m["name"] for m in current_models]

    if hasattr(args, 'template') and args.template:
        # Use template
        if args.template not in list_available_templates():
            print_error(f"Template '{args.template}' not found.")
            print_info("Available templates:")
            for template in list_available_templates():
                print(f"  • {template}")
            return

        # Create model from template
        try:
            model = create_model_from_template(
                args.template,
                args.name,
                args.description,
                args.repo if hasattr(args, 'repo') else None
            )

            # Check for name conflicts
            if model["name"] in current_names:
                print_warning(f"Model '{model['name']}' already exists!")
                overwrite = input("Overwrite existing model? (y/N): ").lower().strip()
                if overwrite != 'y':
                    print_info("Model addition cancelled.")
                    return

                # Remove existing model
                current_models = [m for m in current_models if m["name"] != model["name"]]

        except Exception as e:
            print_error(f"Failed to create model from template: {e}")
            return

    elif hasattr(args, 'repo') and args.repo:
        # Create custom Hugging Face model
        if not args.name:
            print_error("Model name is required when using --repo")
            return

        if args.name in current_names:
            print_warning(f"Model '{args.name}' already exists!")
            overwrite = input("Overwrite existing model? (y/N): ").lower().strip()
            if overwrite != 'y':
                print_info("Model addition cancelled.")
                return

            # Remove existing model
            current_models = [m for m in current_models if m["name"] != args.name]

        # Determine model type from repo name or ask user
        model_type = "stt"  # default
        repo_lower = args.repo.lower()

        if any(keyword in repo_lower for keyword in ["tts", "speech", "bark", "tacotron", "fastspeech"]):
            model_type = "tts"
        elif any(keyword in repo_lower for keyword in ["whisper", "wav2vec", "hubert", "stt", "asr"]):
            model_type = "stt"

        if hasattr(args, 'type') and args.type:
            model_type = args.type

        # Create model configuration
        model = {
            "name": args.name,
            "type": model_type,
            "description": args.description or f"Custom {model_type.upper()} model from Hugging Face",
            "source": "huggingface",
            "huggingface_repo": args.repo,
            "license": getattr(args, 'license', "MIT"),
            "size_mb": getattr(args, 'size_mb', 500),
            "requirements": ["transformers", "torch"],
            "tags": getattr(args, 'tags', ["custom", "huggingface"]).split(",") if hasattr(args, 'tags') else ["custom", "huggingface"]
        }

    else:
        print_error("Either --template or --repo is required")
        return

    # Validate the model
    if model["source"] == "huggingface":
        warnings = validate_model_for_huggingface(model)
        if warnings:
            print_warning("Model validation warnings:")
            for warning in warnings:
                print(f"  • {warning}")

    # Add model to list
    current_models.append(model)

    # Save configuration
    metadata = get_config_metadata()
    metadata["last_updated"] = "2024-12-19"  # Update timestamp

    if save_models_config(current_models, metadata):
        print_success(f"✅ Model '{model['name']}' added successfully!")
        print_info(f"📝 Type: {model['type'].upper()}")
        print_info(f"🔗 Source: {model['source']}")
        if "huggingface_repo" in model:
            print_info(f"📦 Repo: {model['huggingface_repo']}")
        print_info(f"📏 Size: {model['size_mb']}MB")

        # Suggest next steps
        print("\n💡 Next steps:")
        print(f"  1. Test with: ollamaaudio list")
        print(f"  2. Run server: ollamaaudio run {model['name']} --port 8000")
        if model["type"] == "stt":
            print(f"  3. Test API: curl -X POST 'http://localhost:8000/transcribe' -F 'file=@audio.wav'")
        else:
            print(f"  3. Test API: curl -X POST 'http://localhost:8000/synthesize' -H 'Content-Type: application/json' -d '{{\"text\": \"Hello world\"}}'")

    else:
        print_error("❌ Failed to save model configuration")

def handle_list_templates(args):
    """Handles the 'list-templates' command."""
    print_header()
    print_success("🎯 Available OllamaAudio Model Templates:")

    templates = list_available_templates()
    popular = get_popular_models()

    print(f"\n📚 All Templates ({len(templates)}):")
    print("-" * 50)

    for name in templates:
        template = get_model_template(name)
        if template:
            print(f"📦 {name}")
            print(f"   Type: {template['type'].upper()}")
            print(f"   Size: {template['size_mb']}MB")
            print(f"   Description: {template['description']}")
            if "tags" in template:
                print(f"   Tags: {', '.join(template['tags'])}")
            print()

    print("🚀 Popular Models (Ready to use):")
    print("-" * 40)

    for model in popular:
        print(f"⭐ {model['name']}")
        print(f"   {model['description']}")
        print(f"   Template: {model.get('tags', [''])[0] if model.get('tags') else 'custom'}")
        print()

    print("💡 Usage examples:")
    print("  ollamaaudio add-model --template whisper_stt --name my-whisper")
    print("  ollamaaudio add-model --repo openai/whisper-medium --name whisper-med --type stt")

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
  ollamaaudio run whisper-tiny-hf     # Run Hugging Face model server
  ollamaaudio ps                      # Show running servers
  ollamaaudio stt audio.wav           # Transcribe audio file (shows model details)
  ollamaaudio tts "Hello world"       # Generate speech (shows engine info)
  ollamaaudio cache info              # Check cache status
  ollamaaudio status                  # Check system status

Model Management:
  ollamaaudio list-templates          # See available model templates
  ollamaaudio add-model --template whisper_stt --name my-whisper
  ollamaaudio add-model --repo openai/whisper-medium --name whisper-med --type stt
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

    # PS command
    parser_ps = subparsers.add_parser("ps", help="Show running OllamaAudio processes and servers.")
    parser_ps.set_defaults(func=handle_ps)

    # Add model command
    parser_add_model = subparsers.add_parser("add-model", help="Add a new model to OllamaAudio.")
    add_model_group = parser_add_model.add_mutually_exclusive_group(required=True)
    add_model_group.add_argument("--template", help="Use a model template.")
    add_model_group.add_argument("--repo", help="Hugging Face repository (org/model).")
    parser_add_model.add_argument("--name", required=True, help="Name for the new model.")
    parser_add_model.add_argument("--description", help="Description of the model.")
    parser_add_model.add_argument("--type", choices=["stt", "tts"], help="Model type (auto-detected from repo if not specified).")
    parser_add_model.add_argument("--license", default="MIT", help="Model license.")
    parser_add_model.add_argument("--size-mb", type=int, default=500, help="Approximate model size in MB.")
    parser_add_model.add_argument("--tags", help="Comma-separated tags for the model.")
    parser_add_model.set_defaults(func=handle_add_model)

    # List templates command
    parser_list_templates = subparsers.add_parser("list-templates", help="List available model templates.")
    parser_list_templates.set_defaults(func=handle_list_templates)

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
