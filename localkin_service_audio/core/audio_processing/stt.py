import whisper
import os
from typing import Dict, Any, Optional

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

def transcribe_audio(model_size: str, audio_file_path: str, engine: str = "auto", enable_vad: bool = True) -> str:
    """
    Transcribes an audio file using Whisper models.

    Args:
        model_size: Size of the model (tiny, base, small, medium, large, etc.)
        audio_file_path: Path to the audio file
        engine: Which engine to use - "openai", "faster", or "auto"
        enable_vad: Enable Voice Activity Detection for faster-whisper (default: True)

    Returns:
        Transcribed text or error message
    """
    if not os.path.exists(audio_file_path):
        return f"Error: Audio file not found at {audio_file_path}"

    try:
        # Auto-select engine based on availability and model size
        if engine == "auto":
            # Use faster-whisper for better performance when available
            if FASTER_WHISPER_AVAILABLE and model_size in ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "distil-large-v3"]:
                engine = "faster"
            else:
                engine = "openai"

        if engine == "faster" and FASTER_WHISPER_AVAILABLE:
            return transcribe_with_faster_whisper(model_size, audio_file_path, enable_vad)
        else:
            # Check if VAD was requested but we're using OpenAI Whisper
            if enable_vad and engine != "faster":
                print("⚠️ Warning: Voice Activity Detection (VAD) is not supported with OpenAI Whisper models.")
                print("💡 Tip: Use faster-whisper models (e.g., --model faster-whisper-base) for VAD support.")
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

def get_available_engines() -> Dict[str, bool]:
    """
    Returns availability of transcription engines.
    """
    return {
        "openai_whisper": True,  # Always available
        "faster_whisper": FASTER_WHISPER_AVAILABLE
    }

def get_faster_whisper_models() -> list:
    """
    Returns list of faster-whisper compatible model sizes.
    """
    return ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo", "distil-large-v3"]