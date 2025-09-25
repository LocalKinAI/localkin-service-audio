import argparse
import sys
import os
from pathlib import Path

# Import from the localkin_service_audio package
try:
    from ..core import (
        get_models, find_model, find_models_by_type, save_models_config, get_config_metadata,
        list_local_models, pull_model, run_ollama_model, run_huggingface_model,
        get_cache_info, clear_cache, transcribe_audio, synthesize_speech
    )
    from ..templates import (
        get_model_template, list_available_templates, create_model_from_template,
        validate_model_for_huggingface, suggest_similar_models, get_popular_models
    )
except ImportError:
    # Fallback for direct execution
    from ..core import (
        get_models, find_model, find_models_by_type, save_models_config, get_config_metadata,
        list_local_models, pull_model, run_ollama_model, run_huggingface_model,
        get_cache_info, clear_cache, transcribe_audio, synthesize_speech
    )
    from ..templates import (
        get_model_template, list_available_templates, create_model_from_template,
        validate_model_for_huggingface, suggest_similar_models, get_popular_models
    )

# Version information
__version__ = "0.1.0"

def print_header():
    """Print the localkin-service-audio header."""
    print("🎵 LocalKin Service Audio - Local STT & TTS Model Manager")
    print("=" * 60)

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

    # Sort models: STT first, then TTS
    stt_models = [m for m in supported_models if m.get('type') == 'stt']
    tts_models = [m for m in supported_models if m.get('type') == 'tts']

    print(f"\n{'MODEL':<25} {'TYPE':<6} {'STATUS':<18} {'SOURCE':<15} {'DESCRIPTION'}")
    print("-" * 90)

    # Display STT models first
    for model in stt_models:
        name = model.get('name', 'Unknown')
        model_type = model.get('type', 'N/A')
        source = model.get('source', 'unknown')
        description = model.get('description', 'No description available')

        # Determine status
        status = "N/A"
        if source == "ollama":
            status = "✅ Pulled" if any(m.startswith(name) for m in local_ollama_models) else "⬇️  Not Pulled"
        elif source in ["openai-whisper", "pyttsx3", "faster-whisper"]:
            status = "📦 Local Library"
        elif source == "huggingface":
            # Check if model is cached
            try:
                from ..core import get_cache_info
                cache_info = get_cache_info()
                cached_models = [m["name"] for m in cache_info["cached_models"]]
                status = "✅ Pulled" if name in cached_models else "⬇️  Not Pulled"
            except Exception as e:
                print_warning(f"Cache check failed for {name}: {e}")
                status = "❓ Unknown"

        print(f"{name:<25} {model_type:<6} {status:<18} {source:<15} {description}")

    # Separator line between STT and TTS
    if tts_models:
        print("-" * 90)

    # Display TTS models
    for model in tts_models:
        name = model.get('name', 'Unknown')
        model_type = model.get('type', 'N/A')
        source = model.get('source', 'unknown')
        description = model.get('description', 'No description available')

        # Determine status
        status = "N/A"
        if source == "ollama":
            status = "✅ Pulled" if any(m.startswith(name) for m in local_ollama_models) else "⬇️  Not Pulled"
        elif source in ["openai-whisper", "pyttsx3", "faster-whisper"]:
            status = "📦 Local Library"
        elif source == "huggingface":
            # Check if model is cached
            try:
                from ..core import get_cache_info
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
        print_info("Use 'kin audio models' to see available models.")
        return

    source = model_info.get("source")
    huggingface_repo = model_info.get("huggingface_repo")

    success = pull_model(model_name, source, huggingface_repo)
    if success:
        print_success(f"Successfully pulled model: {model_name}")
    else:
        print_error(f"Failed to pull model: {model_name}")

def handle_transcribe(args):
    """Handles the 'audio transcribe' command."""
    print_header()

    input_file = args.input_file
    model = getattr(args, 'model', 'whisper')
    model_size = getattr(args, 'model_size', 'base')
    engine = getattr(args, 'engine', 'auto')

    if not os.path.exists(input_file):
        print_error(f"Input file not found: {input_file}")
        return

    # Determine the actual model to use
    if model != 'whisper':
        # Specific model name provided (e.g., faster-whisper-tiny)
        actual_model = model
        model_display_name = model
    else:
        # Default whisper model with size
        actual_model = model_size
        model_display_name = f"whisper-{model_size}"

    # Show detailed model information
    print_info(f"🎵 Transcribing audio file: {input_file}")
    print_info(f"🤖 Using model: {model_display_name}")

    # Determine which engine will be used
    from ..core.audio_processing.stt import get_available_engines
    engines = get_available_engines()

    # Auto-select engine based on model type
    if engine == "auto":
        if model.startswith("faster-whisper") or (engines["faster_whisper"] and actual_model in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "distil-large-v3"]):
            actual_engine = "faster-whisper (auto-selected for speed)"
        else:
            actual_engine = "OpenAI Whisper (fallback)"
    elif engine == "faster":
        if engines["faster_whisper"]:
            actual_engine = "faster-whisper"
        else:
            print_error("faster-whisper is not available. Install with: pip install faster-whisper")
            return
    else:
        actual_engine = "OpenAI Whisper"

    print_info(f"⚡ Engine: {actual_engine}")

    # Get model details
    model_details = {
        "tiny": {"size": "39MB", "speed": "32x", "quality": "Basic"},
        "base": {"size": "74MB", "speed": "16x", "quality": "Good"},
        "small": {"size": "244MB", "speed": "8x", "quality": "High"},
        "medium": {"size": "769MB", "speed": "4x", "quality": "Very High"},
        "large": {"size": "1550MB", "speed": "1x", "quality": "Excellent"},
        "large-v2": {"size": "3000MB", "speed": "1x", "quality": "Excellent"},
        "large-v3": {"size": "3000MB", "speed": "1x", "quality": "Excellent"},
        "turbo": {"size": "1500MB", "speed": "2x", "quality": "Very High"},
        "distil-large-v3": {"size": "1500MB", "speed": "2x", "quality": "Very High"}
    }

    if model_size in model_details:
        details = model_details[model_size]
        print_info(f"📊 Model details: {details['size']} | {details['speed']} speed | {details['quality']} quality")
    else:
        print_warning(f"Unknown model size: {model_size}")

    print_info("🔄 Processing audio...")

    try:
        transcription = transcribe_audio(actual_model, input_file, engine)
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


def handle_listen(args):
    """Handles the 'audio listen' command - real-time STT/TTS loop."""
    print_header()

    model_size = getattr(args, 'model_size', 'base')
    enable_tts = getattr(args, 'tts', False)

    print_info("🎧 Starting real-time STT/TTS loop...")
    print_info(f"🤖 Using Whisper model: {model_size}")

    if enable_tts:
        print_info("🔊 Text-to-speech output enabled")
    else:
        print_info("🔇 Text-to-speech output disabled (use --tts to enable)")

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

    print_info("🎤 Listening... (Press Ctrl+C to stop)")
    print("-" * 60)

    try:
        import sounddevice as sd
        import numpy as np
        import scipy.io.wavfile as wavfile
        import tempfile
        import time

        # Audio recording parameters
        sample_rate = 16000
        channels = 1
        chunk_duration = 3  # seconds
        chunk_size = sample_rate * chunk_duration

        # Buffer to accumulate audio
        audio_buffer = []

        def audio_callback(indata, frames, time_info, status):
            """Callback function to process audio chunks."""
            if status:
                print_warning(f"Audio status: {status}")

            # Add audio data to buffer
            audio_buffer.extend(indata[:, 0])

            # Process when we have enough audio
            if len(audio_buffer) >= chunk_size:
                # Extract chunk
                chunk = np.array(audio_buffer[:chunk_size], dtype=np.float32)
                audio_buffer[:] = audio_buffer[chunk_size:]  # Remove processed chunk

                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                    wavfile.write(temp_filename, sample_rate, chunk)

                try:
                    # Transcribe the chunk
                    transcription = transcribe_audio(model_size, temp_filename)

                    if not transcription.startswith("Error:") and transcription.strip():
                        print(f"🎵 You said: {transcription}")

                        # Generate TTS if enabled
                        if enable_tts and transcription.strip():
                            try:
                                # Use a simple TTS response
                                response_text = f"I heard: {transcription}"
                                success = synthesize_speech(response_text)
                                if success:
                                    print(f"🔊 TTS: {response_text}")
                            except Exception as tts_e:
                                print_warning(f"TTS failed: {tts_e}")

                except Exception as e:
                    print_warning(f"Transcription failed: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass

        # Start audio stream
        with sd.InputStream(callback=audio_callback,
                           channels=channels,
                           samplerate=sample_rate,
                           blocksize=int(sample_rate * 0.1)):  # 100ms blocks
            print_info("🎤 Real-time listening active. Speak into your microphone...")
            print_info("💡 Say something and the system will transcribe it in real-time")

            while True:
                time.sleep(0.1)  # Keep the main thread alive

    except KeyboardInterrupt:
        print("\n\n⚠️  Real-time listening stopped by user.")
    except ImportError as e:
        print_error(f"Missing required audio libraries: {e}")
        print_info("Install with: pip install sounddevice scipy")
    except Exception as e:
        print_error(f"❌ Real-time listening failed: {e}")

def handle_tts(args):
    """Handles the 'tts' command."""
    print_header()

    text = args.text
    output = args.output
    model_name = getattr(args, 'model', 'native')

    # Show detailed model information
    print_info("🔊 Synthesizing speech...")

    # Get model information
    model_info = find_model(model_name)
    if not model_info:
        print_error(f"Model '{model_name}' not found in configuration.")
        print_info("Use 'kin audio models' to see available models.")
        print_info("For TTS, available models include: native, kokoro-82m, speecht5-tts, bark-small")
        return

    source = model_info.get('source', 'unknown')
    model_type = model_info.get('type', 'unknown')

    # Show model information
    print_info(f"🤖 Using TTS model: {model_name}")
    print_info(f"🔗 Source: {source}")
    if source == 'huggingface':
        repo = model_info.get('huggingface_repo', 'Unknown')
        print_info(f"📦 Repository: {repo}")
    print_info(f"📏 Size: {model_info.get('size_mb', 'Unknown')}MB")

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
        # Route to appropriate TTS implementation based on source
        if source == 'pyttsx3' or model_name == 'native':
            # Use native OS TTS
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

        elif source == 'huggingface':
            # Use Hugging Face TTS models
            success = synthesize_huggingface_tts(model_name, text, output)
            if success:
                if output:
                    print_success(f"✅ Speech synthesized and saved to: {output}")
                else:
                    print_success("✅ Speech synthesized successfully!")
            else:
                print_error("❌ Speech synthesis failed.")

        elif source == 'ollama':
            # Use Ollama TTS models (future implementation)
            print_warning("Ollama TTS models not yet implemented in CLI.")
            print_info("Use 'kin audio run <model_name> --port 8000' to start an API server instead.")
            return

        else:
            print_error(f"Unsupported TTS source: {source}")
            return

    except Exception as e:
        print_error(f"❌ Speech synthesis failed: {e}")
        print_info("Make sure you have the required dependencies installed:")
        if model_name == 'kokoro-82m':
            print_info("  pip install kokoro>=0.9.2 soundfile")
        elif 'speecht5' in model_name:
            print_info("  pip install transformers torch torchaudio")
        elif 'bark' in model_name:
            print_info("  pip install transformers torch")

def synthesize_huggingface_tts(model_name: str, text: str, output_path: str = None) -> bool:
    """Synthesize speech using Hugging Face TTS models."""
    try:
        from ..core.config import find_model

        model_info = find_model(model_name)
        if not model_info or model_info.get("source") != "huggingface":
            print_error(f"Model {model_name} not found or not a Hugging Face model")
            return False

        repo_id = model_info.get("huggingface_repo")
        if not repo_id:
            print_error(f"No Hugging Face repo specified for {model_name}")
            return False

        # Handle different TTS model types
        if "kokoro" in model_name.lower():
            # Kokoro TTS implementation
            print_info("🔧 Initializing Kokoro TTS...")
            try:
                print_info("📚 Importing kokoro modules...")
                from kokoro import KPipeline
                import soundfile as sf
                import numpy as np
                print_success("✅ Kokoro modules imported successfully")
            except ImportError as e:
                print_error("kokoro package is required for Kokoro models.")
                print_info("Install with: pip install kokoro>=0.9.2 soundfile")
                print_warning(f"Import error: {e}")
                return False
            except Exception as e:
                print_error("Failed to import kokoro dependencies.")
                print_warning(f"This may be due to missing system libraries (_lzma). Error: {e}")
                print_info("Try reinstalling Python with lzma support:")
                print_info("  pyenv install 3.10.0  # with proper system dependencies")
                print_info("Or use alternative TTS models: native, speecht5-tts")
                return False

            try:
                print_info("🎯 Creating Kokoro pipeline...")
                # Kokoro supports multiple languages - default to English ('a')
                import time
                import threading

                # Create a timeout mechanism for pipeline creation
                pipeline_created = [False]
                pipeline_instance = [None]
                creation_error = [None]

                def create_pipeline():
                    try:
                        import os
                        from kokoro import KModel

                        # Find the cached model path
                        cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M")
                        if os.path.exists(cache_base):
                            # Find the actual snapshot directory
                            snapshots_dir = os.path.join(cache_base, "snapshots")
                            if os.path.exists(snapshots_dir):
                                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                                if snapshot_dirs:
                                    snapshot_dir = os.path.join(snapshots_dir, snapshot_dirs[0])
                                    model_path = os.path.join(snapshot_dir, "kokoro-v1_0.pth")
                                    config_path = os.path.join(snapshot_dir, "config.json")

                                    if os.path.exists(model_path) and os.path.exists(config_path):
                                        print_info("📁 Loading cached Kokoro model directly...")
                                        # Create KModel from local files
                                        kmodel = KModel(model=model_path, config=config_path)
                                        pipeline_instance[0] = KPipeline(lang_code='a', model=kmodel)
                                        pipeline_created[0] = True
                                        return

                        # Fallback: try normal loading (may download)
                        print_info("⬇️ Downloading/loading Kokoro model...")
                        pipeline_instance[0] = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
                        pipeline_created[0] = True
                    except Exception as e:
                        creation_error[0] = e
                        pipeline_created[0] = False

                # Start pipeline creation in a separate thread
                pipeline_thread = threading.Thread(target=create_pipeline)
                pipeline_thread.daemon = True
                pipeline_thread.start()

                # Wait for pipeline creation with timeout
                start_time = time.time()
                timeout_seconds = 30  # 30 second timeout - if it takes longer, recommend API server

                print_info("⏳ Loading Kokoro model (using cached files)...")
                while not pipeline_created[0] and (time.time() - start_time) < timeout_seconds:
                    if creation_error[0]:
                        raise creation_error[0]
                    elapsed = int(time.time() - start_time)
                    if elapsed % 5 == 0 and elapsed > 0:  # Progress update every 5 seconds
                        print_info(f"⏳ Still loading... ({elapsed}s)")
                    time.sleep(0.1)

                if not pipeline_created[0]:
                    if creation_error[0]:
                        raise creation_error[0]
                    else:
                        raise TimeoutError("Kokoro model loading timed out - use API server instead")

                pipeline = pipeline_instance[0]
                creation_time = time.time() - start_time
                print_success(f"✅ Kokoro pipeline created successfully in {creation_time:.1f} seconds")

            except TimeoutError:
                print_error("❌ Kokoro CLI loading timed out")
                print_info("💡 For better Kokoro TTS performance, use the API server:")
                print_info("  uv run kin audio run kokoro-82m --port 8003")
                print_info("  curl -X POST 'http://localhost:8003/synthesize' \\")
                print_info("       -H 'Content-Type: application/json' \\")
                print_info("       -d '{\"text\": \"Hello world\"}' \\")
                print_info("       --output speech.wav")
                print_info("")
                print_info("Alternative: Use native TTS immediately:")
                print_info("  uv run kin audio tts 'Hello world' --model native")
                return False
            except Exception as e:
                print_error(f"❌ Failed to create Kokoro pipeline: {e}")
                print_warning("This might be due to network issues or model download problems")
                print_info("Try using a different TTS model:")
                print_info("  uv run kin audio tts 'text' --model native")
                return False

                # Default voice for Kokoro
                voice = 'af_heart'  # You can make this configurable later

                # Generate speech using Kokoro
                print_info(f"🎤 Using voice: {voice}")
                print_info("🎵 Generating audio segments...")

                generator = pipeline(
                    text,
                    voice=voice,
                    speed=1.0,
                )

                # Collect all audio segments
                audio_segments = []
                segment_count = 0

                for gs, ps, audio in generator:
                    audio_segments.append(audio)
                    segment_count += 1
                    print_info(f"📊 Generated segment {segment_count} (shape: {audio.shape})")

                print_info(f"📈 Total segments generated: {len(audio_segments)}")

                # Concatenate all audio segments
                if audio_segments:
                    final_audio = np.concatenate(audio_segments)
                    print_info(f"🔗 Concatenated audio shape: {final_audio.shape}")
                    print_info(f"🎵 Audio duration: {len(final_audio) / 24000:.2f} seconds")
                else:
                    print_error("❌ No audio segments were generated")
                    final_audio = np.array([])
                    return False

                if output_path:
                    # Save to specified file
                    print_info(f"💾 Saving to: {output_path}")
                    sf.write(output_path, final_audio, 24000)  # Kokoro uses 24kHz
                    print_success(f"✅ Audio saved successfully!")
                else:
                    # Play audio directly (this would require additional audio playback libraries)
                    print_warning("Direct audio playback not implemented for Kokoro models.")
                    print_info("Please specify an output file with --output")
                    return False

                return True

            except Exception as e:
                print_error(f"Kokoro TTS synthesis failed: {e}")
                return False

        elif "speecht5" in model_name.lower():
            # SpeechT5 TTS implementation
            try:
                from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                import torch
                import torchaudio
                import numpy as np
            except ImportError:
                print_error("transformers, torch, and torchaudio are required for SpeechT5 models.")
                print_info("Install with: pip install transformers torch torchaudio")
                return False

            try:
                processor = SpeechT5Processor.from_pretrained(repo_id)
                model = SpeechT5ForTextToSpeech.from_pretrained(repo_id)
                vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

                inputs = processor(text=text, return_tensors="pt")

                # Generate speech
                with torch.no_grad():
                    speech = model.generate_speech(
                        inputs["input_ids"],
                        speaker_embeddings=None,  # Use default speaker
                        vocoder=vocoder
                    )

                if output_path:
                    # Save to specified file
                    torchaudio.save(output_path, speech, 16000)
                else:
                    # Play audio directly (would need audio playback library)
                    print_warning("Direct audio playback not implemented for SpeechT5 models.")
                    print_info("Please specify an output file with --output")
                    return False

                return True

            except Exception as e:
                print_error(f"SpeechT5 TTS synthesis failed: {e}")
                return False

        elif "bark" in model_name.lower():
            # Bark TTS implementation (placeholder for now)
            print_warning("Bark TTS implementation not yet fully implemented in CLI.")
            print_info("Use 'kin audio run bark-small --port 8000' to start an API server instead.")
            return False

        else:
            # Generic TTS pipeline (fallback)
            try:
                from transformers import pipeline
                import torch

                device = 0 if torch.cuda.is_available() else -1
                pipe = pipeline(
                    "text-to-speech",
                    model=repo_id,
                    device=device,
                )

                result = pipe(text)

                # Handle the result (implementation depends on specific model)
                print_warning(f"Generic TTS pipeline used for {model_name}. Results may vary.")
                print_info("For best results, use model-specific implementations.")

                if output_path:
                    # Try to save the result (this is a placeholder implementation)
                    # The actual implementation would depend on the model's output format
                    print_error(f"Generic TTS saving not implemented for {model_name}")
                    return False
                else:
                    print_warning("Direct audio playback not available for generic TTS models.")
                    return False

            except Exception as e:
                print_error(f"Generic TTS synthesis failed: {e}")
                return False

    except Exception as e:
        print_error(f"❌ Hugging Face TTS synthesis failed: {e}")
        return False

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
            print_info("Local STT models don't require a server - use 'kin audio transcribe' command instead.")
        elif model_type == 'tts':
            print_info("Local TTS models don't require a server - use 'kin audio tts' command instead.")
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
        from ..core import get_cache_info
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

    print_info("Checking for running LocalKin Service Audio processes...")

    running_servers = []

    # Check common ports for LocalKin Service Audio servers
    common_ports = [8000, 8001, 8002, 8003, 8004, 8005]

    for port in common_ports:
        try:
            # Check if port is in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()

            if result == 0:
                # Port is open, check if it's a LocalKin Service Audio server
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/", timeout=2)
                    if response.status_code == 200 and "LocalKin Service Audio" in response.text:
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
                    # Port is open but not a LocalKin Service Audio server
                    continue
        except:
            continue

    if running_servers:
        print_success(f"Found {len(running_servers)} running LocalKin Service Audio server(s):")
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
        print_info("No running LocalKin Service Audio servers found.")
        print("💡 Start a server with: kin audio run <model_name> --port <port>")

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
        print(f"  1. Test with: kin audio models")
        print(f"  2. Run server: kin audio run {model['name']} --port 8000")
        if model["type"] == "stt":
            print(f"  3. Test API: curl -X POST 'http://localhost:8000/transcribe' -F 'file=@audio.wav'")
        else:
            print(f"  3. Test API: curl -X POST 'http://localhost:8000/synthesize' -H 'Content-Type: application/json' -d '{{\"text\": \"Hello world\"}}'")

    else:
        print_error("❌ Failed to save model configuration")

def handle_list_templates(args):
    """Handles the 'list-templates' command."""
    print_header()
    print_success("🎯 Available LocalKin Service Audio Model Templates:")

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
    print("  kin audio add-model --template whisper_stt --name my-whisper")
    print("  kin audio add-model --repo openai/whisper-medium --name whisper-med --type stt")

def handle_version(args):
    """Handles the 'version' command."""
    print_header()
    print(f"🎵 LocalKin Service Audio version {__version__}")
    print(f"📍 Location: {Path(__file__).parent.absolute()}")

def handle_web(args):
    """Handles the 'web' command to launch the web-based user interface."""
    print_header()

    try:
        from ..api import create_app
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from pathlib import Path
        import uvicorn
        import webbrowser
        import threading
        import time

        print_info("🚀 Starting LocalKin Service Audio Web Interface...")
        print_info(f"📍 Web UI will be available at: http://{args.host}:{args.port}")

        # Create a dedicated web UI app
        app = FastAPI(
            title="LocalKin Service Audio Web Interface",
            description="Modern web interface for LocalKin Service Audio processing",
            version="1.0.0"
        )

        # Include UI routes
        try:
            from ..ui import create_ui_router
            ui_router = create_ui_router()
            app.include_router(ui_router, prefix="", tags=["ui"])

            # Mount static files
            ui_static_path = Path(__file__).parent.parent / "ui" / "static"
            if ui_static_path.exists():
                app.mount("/ui/static", StaticFiles(directory=str(ui_static_path)), name="ui-static")

        except ImportError:
            print_warning("Web UI components not available, running basic interface")

        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open(f"http://{args.host}:{args.port}")
            except:
                pass  # Silently fail if browser can't be opened

        # Start browser opener in background
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

        print_info("🌐 Web interface starting up...")
        print_info("💡 Use Ctrl+C to stop the server")

        # Start the server
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )

    except ImportError as e:
        print_error(f"Failed to import required modules: {e}")
        print_error("Make sure you have installed LocalKin Service Audio with web dependencies:")
        print_error("  uv sync --extra web")
    except Exception as e:
        print_error(f"Failed to start web interface: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="🎵 LocalKin Service Audio - Local STT & TTS Model Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kin audio listen                    # Start real-time STT/TTS loop
  kin audio transcribe audio.wav      # Transcribe a single audio file
  kin audio transcribe audio.wav --model_size large  # Use specific model size

Web Interface:
  kin web                             # Launch web-based user interface
  kin web --port 3000                 # Launch on specific port

Model Management:
  kin audio models                    # List all available models
  kin audio pull whisper-large-v3     # Pull an Ollama model
  kin audio pull whisper-tiny-hf      # Pull from Hugging Face
  kin audio run whisper-tiny-hf       # Run Hugging Face model server
  kin audio status                    # Check system status
  kin audio cache info                # Check cache status
        """
    )

    parser.add_argument('-v', '--version', action='version', version=f'LocalKin Service Audio {__version__}')

    # Create main subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Audio command (main functionality)
    parser_audio = subparsers.add_parser("audio", help="Audio processing commands")
    audio_subparsers = parser_audio.add_subparsers(dest="audio_command", help="Audio subcommands")

    # Web UI command
    parser_web = subparsers.add_parser("web", help="Launch web-based user interface")
    parser_web.add_argument("--host", default="0.0.0.0", help="Host to bind the web server to")
    parser_web.add_argument("--port", "-p", type=int, default=8080, help="Port for the web interface")
    parser_web.set_defaults(func=handle_web)

    # Listen command (real-time STT/TTS loop)
    parser_listen = audio_subparsers.add_parser("listen", help="Start real-time STT/TTS loop")
    parser_listen.add_argument("--model", "-m", default="whisper", help="The STT model to use.")
    parser_listen.add_argument("--model_size", default="base", help="The size of the Whisper model to use.")
    parser_listen.add_argument("--tts", action="store_true", help="Enable text-to-speech output.")
    parser_listen.set_defaults(func=handle_listen)

    # Transcribe command
    parser_transcribe = audio_subparsers.add_parser("transcribe", help="Transcribe a single audio file")
    parser_transcribe.add_argument("input_file", help="Path to the audio file to transcribe.")
    parser_transcribe.add_argument("--model", "-m", default="whisper", help="The STT model to use.")
    parser_transcribe.add_argument("--model_size", default="base", help="The size of the Whisper model to use.")
    parser_transcribe.add_argument("--engine", choices=["auto", "openai", "faster"], default="auto",
                                  help="Transcription engine to use: auto (default), openai, or faster.")
    parser_transcribe.set_defaults(func=handle_transcribe)

    # Models command
    parser_models = audio_subparsers.add_parser("models", help="List all available models")
    parser_models.add_argument("--verbose", "-v", action="store_true", help="Show detailed model information.")
    parser_models.set_defaults(func=handle_list)

    # Pull command
    parser_pull = audio_subparsers.add_parser("pull", help="Pull a model from Ollama or Hugging Face.")
    parser_pull.add_argument("model_name", help="The name of the model to pull.")
    parser_pull.set_defaults(func=handle_pull)

    # Run command
    parser_run = audio_subparsers.add_parser("run", help="Run a model server.")
    parser_run.add_argument("model_name", help="The name of the model to run.")
    parser_run.add_argument("--port", "-p", type=int, default=8000, help="Port for the model server.")
    parser_run.set_defaults(func=handle_run)

    # TTS command
    parser_tts = audio_subparsers.add_parser("tts", help="Perform Text-to-Speech using local models.")
    parser_tts.add_argument("text", help="The text to synthesize.")
    parser_tts.add_argument("--output", "-o", help="Path to save the output audio file.")
    parser_tts.add_argument("--model", "-m", default="native", help="The TTS model to use.")
    parser_tts.set_defaults(func=handle_tts)

    # Status command
    parser_status = audio_subparsers.add_parser("status", help="Check system and model status.")
    parser_status.set_defaults(func=handle_status)

    # Cache command
    parser_cache = audio_subparsers.add_parser("cache", help="Manage model cache.")
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
    parser_ps = audio_subparsers.add_parser("ps", help="Show running LocalKin Service Audio processes and servers.")
    parser_ps.set_defaults(func=handle_ps)

    # Add model command
    parser_add_model = audio_subparsers.add_parser("add-model", help="Add a new model to LocalKin Service Audio.")
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
    parser_list_templates = audio_subparsers.add_parser("list-templates", help="List available model templates.")
    parser_list_templates.set_defaults(func=handle_list_templates)

    # Version command
    parser_version = audio_subparsers.add_parser("version", help="Show version information.")
    parser_version.set_defaults(func=handle_version)

    # Set default audio command
    parser_audio.set_defaults(audio_command="transcribe")

    # If no command provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    # Handle nested command structure
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            sys.exit(1)
    else:
        # Handle case where audio command is provided but no subcommand
        if args.command == "audio" and not hasattr(args, 'audio_command'):
            parser_audio.print_help()
            return

if __name__ == "__main__":
    main()
