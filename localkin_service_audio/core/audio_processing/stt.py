import whisper
import os
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    from .whisper_cpp import WhisperCppSTT, transcribe_with_whisper_cpp, get_whisper_cpp_engines
    WHISPER_CPP_AVAILABLE = True
except ImportError:
    WHISPER_CPP_AVAILABLE = False
    WhisperCppSTT = None
    transcribe_with_whisper_cpp = None
    get_whisper_cpp_engines = None

def transcribe_audio(model_size: str, audio_file_path: str, engine: str = "auto", enable_vad: bool = True, **kwargs) -> str:
    """
    Transcribes an audio file using Whisper models.

    Args:
        model_size: Size of the model (tiny, base, small, medium, large, etc.)
        audio_file_path: Path to the audio file
        engine: Which engine to use - "openai", "faster", "whisper-cpp", or "auto"
        enable_vad: Enable Voice Activity Detection for faster-whisper (default: True)
        **kwargs: Additional arguments passed to the transcription engine

    Returns:
        Transcribed text or error message
    """
    if not os.path.exists(audio_file_path):
        return f"Error: Audio file not found at {audio_file_path}"

    try:
        # Auto-select engine based on availability and model size
        if engine == "auto":
            # Priority: whisper-cpp (fastest) > faster-whisper > openai-whisper
            if WHISPER_CPP_AVAILABLE and model_size in ["tiny", "base", "small", "medium", "large"]:
                engine = "whisper-cpp"
            elif FASTER_WHISPER_AVAILABLE and model_size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "distil-large-v3"]:
                engine = "faster"
            else:
                engine = "openai"

        if engine == "whisper-cpp" and WHISPER_CPP_AVAILABLE:
            return transcribe_with_whisper_cpp_engine(model_size, audio_file_path, **kwargs)
        elif engine == "faster" and FASTER_WHISPER_AVAILABLE:
            return transcribe_with_faster_whisper(model_size, audio_file_path, enable_vad)
        else:
            # Check if VAD was requested but we're using OpenAI Whisper
            if enable_vad and engine not in ["faster", "whisper-cpp"]:
                print("⚠️ Warning: Voice Activity Detection (VAD) is not supported with OpenAI Whisper models.")
                print("💡 Tip: Use faster-whisper or whisper-cpp models for VAD support.")
            return transcribe_with_openai_whisper(model_size, audio_file_path)

    except Exception as e:
        return f"An unexpected error occurred during transcription: {e}"

def transcribe_with_openai_whisper(model_size: str, audio_file_path: str) -> str:
    """
    Transcribes using the original OpenAI Whisper implementation.
    """
    try:
        print(f"Loading OpenAI Whisper model '{model_size}'... (This might download the model on first use)")
        model = whisper.load_model(model_size)

        print(f"Transcribing {audio_file_path} with OpenAI Whisper...")
        result = model.transcribe(audio_file_path, fp16=False)  # fp16=False for CPU compatibility

        transcribed_text = result["text"]
        print("Transcription complete.")
        return transcribed_text

    except Exception as e:
        return f"OpenAI Whisper transcription failed: {e}"

def transcribe_with_faster_whisper(model_size: str, audio_file_path: str, enable_vad: bool = True) -> str:
    """
    Transcribes using faster-whisper (CTranslate2 implementation).
    Up to 4x faster than OpenAI Whisper.

    Args:
        model_size: The model size to use
        audio_file_path: Path to the audio file
        enable_vad: Enable Voice Activity Detection (default: True)
    """
    if not FASTER_WHISPER_AVAILABLE:
        return "Error: faster-whisper is not available. Please install it with: pip install faster-whisper"

    try:
        print(f"Loading faster-whisper model '{model_size}'... (This might download the model on first use)")

        # Handle both size-based and model name-based inputs
        if model_size.startswith("faster-whisper-"):
            # Extract the actual model size from the model name
            # e.g., "faster-whisper-tiny" -> "tiny"
            actual_model_size = model_size.replace("faster-whisper-", "")
        else:
            # Legacy size-based mapping for backward compatibility
            model_size_map = {
                "large-v3": "large-v3",
                "large-v2": "large-v2",
                "large": "large-v2",  # Default to v2 for "large"
                "medium": "medium",
                "small": "small",
                "base": "base",
                "tiny": "tiny",
                "turbo": "turbo",
                "distil-large-v3": "distil-large-v3"
            }
            actual_model_size = model_size_map.get(model_size, model_size)

        # faster-whisper only supports CPU and CUDA (not MPS on Mac)
        # Note: MPS acceleration is not available for faster-whisper
        device = "cpu"
        compute_type = "int8"

        print(f"Using device: {device} (faster-whisper doesn't support MPS/CUDA acceleration)")
        model = WhisperModel(actual_model_size, device=device, compute_type=compute_type)

        # Check audio file duration to choose optimal inference method
        import wave
        import contextlib

        try:
            with contextlib.closing(wave.open(audio_file_path, 'r')) as f:
                duration = f.getnframes() / float(f.getframerate())
        except:
            duration = 0  # Fallback if can't determine duration

        # Use batched inference for long audio files (>5 minutes) where it shines
        # For shorter files, regular inference is faster due to less overhead
        if duration > 300:  # 5 minutes
            from faster_whisper import BatchedInferencePipeline
            batched_model = BatchedInferencePipeline(model=model)
            print(f"Transcribing {audio_file_path} with faster-whisper (batched inference for long audio)...")

            segments, info = batched_model.transcribe(
                audio_file_path,
                batch_size=4,  # Balanced batch size for CPU performance
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=enable_vad,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500) if enable_vad else None
            )
        else:
            print(f"Transcribing {audio_file_path} with faster-whisper (standard inference)...")
            # Use regular inference for shorter files
            segments, info = model.transcribe(
                audio_file_path,
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=enable_vad,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500) if enable_vad else None
            )

        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        # Combine all segments into full text
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text

        print("Transcription complete.")
        return transcribed_text.strip()

    except Exception as e:
        return f"Faster-whisper transcription failed: {e}"

def transcribe_with_whisper_cpp_engine(model_size: str, audio_file_path: str, **kwargs) -> str:
    """
    Transcribes using whisper.cpp (C/C++ implementation).
    Extremely fast and efficient.

    Args:
        model_size: The model size to use (tiny, base, small, medium, large)
        audio_file_path: Path to the audio file
        **kwargs: Additional arguments for whisper-cli

    Returns:
        Transcribed text
    """
    if not WHISPER_CPP_AVAILABLE:
        return "Error: whisper.cpp is not available. Please install whisper.cpp and ensure whisper-cli is in your PATH."

    try:
        # Map model size to whisper.cpp model name
        model_name_map = {
            "tiny": "whisper-cpp-tiny",
            "base": "whisper-cpp-base",
            "small": "whisper-cpp-small",
            "medium": "whisper-cpp-medium",
            "large": "whisper-cpp-large"
        }

        whisper_cpp_model = model_name_map.get(model_size, model_size)

        # Try to find or download the model
        stt = WhisperCppSTT()

        # Check if it's already a whisper-cpp model name
        if not whisper_cpp_model.startswith("whisper-cpp-"):
            whisper_cpp_model = f"whisper-cpp-{model_size}"

        print(f"Loading whisper.cpp model '{whisper_cpp_model}'...")

        # Try to download the model if not available
        try:
            model_path = stt.download_model(whisper_cpp_model)
        except Exception as download_e:
            return f"Failed to download whisper.cpp model: {download_e}"

        print(f"Transcribing {audio_file_path} with whisper.cpp...")

        # Extract language from kwargs if provided
        language = kwargs.pop('language', None)

        # Transcribe with whisper.cpp
        result = stt.transcribe(
            audio_file_path,
            model_path=model_path,
            language=language,
            output_format="txt",
            **kwargs
        )

        print("Transcription complete.")
        return result

    except Exception as e:
        return f"whisper.cpp transcription failed: {e}"

def get_available_engines() -> Dict[str, bool]:
    """
    Returns availability of transcription engines.
    """
    engines = {
        "openai_whisper": True,  # Always available
        "faster_whisper": FASTER_WHISPER_AVAILABLE,
        "whisper_cpp": WHISPER_CPP_AVAILABLE
    }

    # If whisper.cpp is available, also check if executable is actually found
    if WHISPER_CPP_AVAILABLE and get_whisper_cpp_engines:
        cpp_engines = get_whisper_cpp_engines()
        engines["whisper_cpp"] = cpp_engines.get("whisper_cpp", False)

    return engines

def get_faster_whisper_models() -> list:
    """
    Returns list of faster-whisper compatible model sizes.
    """
    return ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo", "distil-large-v3"]