import whisper
import os
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import pywhispercpp as pw
    WHISPER_CPP_AVAILABLE = True
except ImportError:
    WHISPER_CPP_AVAILABLE = False

# Keep old implementation as fallback
try:
    from .whisper_cpp import WhisperCppSTT, transcribe_with_whisper_cpp, get_whisper_cpp_engines
    WHISPER_CPP_AVAILABLE_OLD = True
except ImportError:
    WHISPER_CPP_AVAILABLE_OLD = False
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
    Transcribes using pywhispercpp (Python bindings for whisper.cpp).
    Extremely fast and efficient.

    Args:
        model_size: The model size to use (tiny, base, small, medium, large)
        audio_file_path: Path to the audio file
        **kwargs: Additional arguments

    Returns:
        Transcribed text
    """
    if not WHISPER_CPP_AVAILABLE:
        # Fallback to old implementation if available
        if WHISPER_CPP_AVAILABLE_OLD and transcribe_with_whisper_cpp:
            return transcribe_with_whisper_cpp(model_size, audio_file_path, **kwargs)
        return "Error: pywhispercpp is not available. Please install it with: pip install pywhispercpp"

    try:
        # pywhispercpp uses model names without .bin extension
        model_filename = model_size

        print(f"Loading pywhispercpp model '{model_filename}'... (This might download the model on first use)")

        # Import the Model class from pywhispercpp
        from pywhispercpp.model import Model

        # Initialize whisper model with pywhispercpp
        model = Model(model_filename)

        print(f"Transcribing {audio_file_path} with pywhispercpp...")

        # Load audio with librosa and resample if needed (Whisper requires 16kHz)
        import numpy as np
        import librosa

        # Load audio file with librosa (handles resampling automatically)
        print(f"Loading audio file: {audio_file_path}")
        audio_array, sample_rate = librosa.load(audio_file_path, sr=16000, mono=True)

        # Extract language from kwargs if provided
        language = kwargs.pop('language', None)

        # Transcribe with pywhispercpp using numpy array
        result = model.transcribe(audio_array, language=language if language else None)

        # Extract text from result - pywhispercpp returns a list of segments
        transcribed_text = ""
        if isinstance(result, list):
            for segment in result:
                if hasattr(segment, 'text'):
                    transcribed_text += segment.text
                elif isinstance(segment, dict) and 'text' in segment:
                    transcribed_text += segment['text']
        elif isinstance(result, dict) and 'text' in result:
            transcribed_text = result['text']
        elif isinstance(result, str):
            transcribed_text = result

        print("Transcription complete.")
        return transcribed_text.strip()

    except Exception as e:
        return f"pywhispercpp transcription failed: {e}"

def get_available_engines() -> Dict[str, bool]:
    """
    Returns availability of transcription engines.
    """
    engines = {
        "openai_whisper": True,  # Always available
        "faster_whisper": FASTER_WHISPER_AVAILABLE,
        "whisper_cpp": WHISPER_CPP_AVAILABLE  # Now uses pywhispercpp
    }

    # Keep old whisper-cpp check as fallback
    if not WHISPER_CPP_AVAILABLE and WHISPER_CPP_AVAILABLE_OLD and get_whisper_cpp_engines:
        cpp_engines = get_whisper_cpp_engines()
        engines["whisper_cpp"] = cpp_engines.get("whisper_cpp", False)

    return engines

def get_faster_whisper_models() -> list:
    """
    Returns list of faster-whisper compatible model sizes.
    """
    return ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo", "distil-large-v3"]