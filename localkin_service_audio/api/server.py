"""
LocalKin Service Audio API Server for Hugging Face models
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from ..core.config import model_registry
from ..ui import create_ui_router
from .transcription_format import (
    FormatSegment,
    SUPPORTED_FORMATS,
    to_markdown,
    to_srt,
    to_vtt,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_model_dict(model_name: str) -> Optional[Dict[str, Any]]:
    """Find a model and return its info as a dict (bridge from legacy API)."""
    reg_model = model_registry.get(model_name)
    if reg_model:
        return {
            "name": reg_model.name,
            "type": reg_model.type.value,
            "source": reg_model.engine,
            "engine": reg_model.engine,
            "huggingface_repo": reg_model.repo_id,
            "model_size": reg_model.model_size,
        }
    return None

# Cache configuration - use settings for LOCALKIN_HOME support
from pathlib import Path
from localkin_service_audio.core.config.settings import _default_home
HF_CACHE_DIR = _default_home() / "cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_info():
    """Gets information about cached models."""
    cache_info = {
        "huggingface_cache": str(HF_CACHE_DIR),
        "cached_models": []
    }

    # Check Hugging Face cache
    if HF_CACHE_DIR.exists():
        for model_dir in HF_CACHE_DIR.iterdir():
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                cache_info["cached_models"].append({
                    "name": model_dir.name,
                    "size_mb": round(size / (1024 * 1024), 2)
                })

    return cache_info

# Global model instances
loaded_models = {}

class TranscriptionRequest(BaseModel):
    audio_path: Optional[str] = None
    language: Optional[str] = None
    task: str = "transcribe"


class TranscriptionSegment(BaseModel):
    """A single transcription segment with timing information."""
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    segments: Optional[List[TranscriptionSegment]] = None

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None

class TTSResponse(BaseModel):
    audio_path: str
    duration: Optional[float] = None

def load_whisper_model(model_name: str):
    """Load a Whisper STT model (HuggingFace or whisper-cpp)."""
    try:
        if "whisper-cpp" in model_name:
            # Use pywhispercpp for whisper-cpp models
            from pywhispercpp.model import Model as WhisperModel
            # Extract size: "whisper-cpp:base" -> "base"
            size = model_name.split(":")[-1] if ":" in model_name else "base"
            logger.info(f"Loading whisper-cpp model: {size}")
            model = WhisperModel(size)
            loaded_models[model_name] = {
                "type": "whisper-cpp",
                "model": model,
            }
            logger.info(f"Successfully loaded whisper-cpp model: {size}")
            return model

        # Check model registry for engine type
        model_info = _find_model_dict(model_name)
        if not model_info:
            raise ValueError(f"Model {model_name} not found")
        engine = model_info.get("engine")

        # Moonshine (ONNX or PyTorch): engine="moonshine"
        if engine == "moonshine":
            from ..core.audio_processing.stt.moonshine_strategy import MoonshineStrategy
            from ..core.types import ModelConfig as MC, ModelType
            strategy = MoonshineStrategy()
            size = model_info.get("model_size", "base")
            mc = MC(name=model_name, type=ModelType.STT, engine="moonshine", model_size=size)
            if not strategy.load(mc):
                raise ValueError(f"Failed to load Moonshine model: {model_name}")
            loaded_models[model_name] = {
                "type": "moonshine",
                "strategy": strategy,
            }
            logger.info(f"Successfully loaded Moonshine: {model_name}")
            return strategy

        # SenseVoice (FunASR): engine="sensevoice"
        if engine == "sensevoice":
            from ..core.audio_processing.stt.sensevoice_strategy import SenseVoiceStrategy
            from ..core.types import ModelConfig as MC, ModelType
            strategy = SenseVoiceStrategy()
            size = model_info.get("model_size", "small")
            mc = MC(name=model_name, type=ModelType.STT, engine="sensevoice", model_size=size)
            if not strategy.load(mc):
                raise ValueError(f"Failed to load SenseVoice model: {model_name}")
            loaded_models[model_name] = {
                "type": "sensevoice",
                "strategy": strategy,
            }
            logger.info(f"Successfully loaded SenseVoice: {model_name}")
            return strategy

        # faster-whisper (CTranslate2): engine="faster-whisper"
        if engine == "faster-whisper":
            from ..core.audio_processing.stt.faster_whisper_strategy import FasterWhisperStrategy
            from ..core.types import ModelConfig as MC, ModelType
            strategy = FasterWhisperStrategy()
            size = model_info.get("model_size", "base")
            mc = MC(name=model_name, type=ModelType.STT, engine="faster-whisper", model_size=size)
            if not strategy.load(mc):
                raise ValueError(f"Failed to load faster-whisper model: {model_name}")
            loaded_models[model_name] = {
                "type": "faster-whisper",
                "strategy": strategy,
            }
            logger.info(f"Successfully loaded faster-whisper: {model_name}")
            return strategy

        # OpenAI Whisper (native): engine="whisper" or source="openai-whisper"
        if engine == "whisper" or model_info.get("source") == "openai-whisper":
            import whisper
            size = model_info.get("model_size", "medium")
            logger.info(f"Loading OpenAI Whisper model: {size}")
            model = whisper.load_model(size)
            loaded_models[model_name] = {
                "type": "openai-whisper",
                "model": model,
            }
            logger.info(f"Successfully loaded OpenAI Whisper: {size}")
            return model

        # HuggingFace transformers pipeline
        from transformers import pipeline
        import torch

        repo_id = model_info.get("huggingface_repo")
        if not repo_id:
            raise ValueError(f"Model {model_name} has no Hugging Face repo")

        logger.info(f"Loading Whisper model: {repo_id}")

        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model=repo_id,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
        )

        loaded_models[model_name] = {
            "type": "whisper",
            "pipeline": pipe,
            "repo_id": repo_id
        }

        logger.info(f"Successfully loaded Whisper model: {model_name}")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load Whisper model {model_name}: {e}")
        raise

def load_tts_model(model_name: str):
    """Load a TTS model."""
    try:
        model_info = _find_model_dict(model_name)
        source = model_info.get("source", "") if model_info else ""
        repo_id = model_info.get("huggingface_repo", "") if model_info else ""

        # Kokoro doesn't need huggingface repo
        if "kokoro" in model_name.lower():
            pass  # handled below
        elif not model_info or source != "huggingface" or not repo_id:
            raise ValueError(f"Model {model_name} not found or not a Hugging Face model")

        logger.info(f"Loading TTS model: {model_name}")

        # Load TTS pipeline based on model type
        if "speecht5" in model_name.lower():
            from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import torch
            import torchaudio

            processor = SpeechT5Processor.from_pretrained(repo_id)
            model = SpeechT5ForTextToSpeech.from_pretrained(repo_id)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            loaded_models[model_name] = {
                "type": "speecht5",
                "processor": processor,
                "model": model,
                "vocoder": vocoder,
                "repo_id": repo_id
            }

        elif "bark" in model_name.lower():
            # Bark models use specific Bark classes
            from transformers import BarkProcessor, BarkModel
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            processor = BarkProcessor.from_pretrained(repo_id)
            model = BarkModel.from_pretrained(repo_id)
            model = model.to(device)

            loaded_models[model_name] = {
                "type": "bark",
                "processor": processor,
                "model": model,
                "repo_id": repo_id,
                "device": device
            }

        elif "kokoro" in model_name.lower():
            # Kokoro models use the kokoro library
            try:
                from kokoro import KPipeline
                import soundfile as sf
            except ImportError as e:
                if "_lzma" in str(e):
                    raise ImportError(
                        "Kokoro TTS requires LZMA compression support which is missing from your Python installation. "
                        "Try using system Python instead: /usr/bin/python3, or reinstall Python with LZMA support."
                    )
                elif "AlbertModel" in str(e):
                    raise ImportError(
                        "Kokoro TTS failed to import required transformers components. "
                        "This may be due to missing LZMA support. Try using system Python: /usr/bin/python3"
                    )
                else:
                    raise ImportError(f"kokoro package is required for Kokoro models. Install with: pip install kokoro>=0.9.2. Error: {e}")

            # Kokoro supports multiple languages - default to English ('a')
            try:
                pipeline = KPipeline(lang_code='a')  # 'a' for American English
            except SystemExit as e:
                # Handle spaCy download failures (kokoro tries to download en_core_web_sm)
                if "pip" in str(e).lower() or "spacy" in str(e).lower():
                    raise RuntimeError(
                        "Kokoro TTS requires spaCy English model (en_core_web_sm) but it failed to download. "
                        "Try installing it manually: "
                        "python -m spacy download en_core_web_sm"
                    )
                else:
                    raise e
            except Exception as e:
                if "_lzma" in str(e):
                    raise RuntimeError(
                        "Kokoro TTS requires LZMA compression support. "
                        "Try using system Python instead: /usr/bin/python3"
                    )
                elif "spacy" in str(e).lower() or "en_core_web_sm" in str(e).lower():
                    raise RuntimeError(
                        "Kokoro TTS requires spaCy English model. "
                        "Install it with: python -m spacy download en_core_web_sm"
                    )
                else:
                    raise RuntimeError(f"Failed to initialize Kokoro TTS pipeline: {e}")

            loaded_models[model_name] = {
                "type": "kokoro",
                "pipelines": {"a": pipeline},  # Lazy-load other languages on demand
                "repo_id": repo_id,
            }

        elif model_name == "xtts-v2":
            # XTTS specific loading
            try:
                from TTS.api import TTS
                import os
                import torch
            except ImportError:
                raise ImportError("TTS package is required for XTTS models. Install with: pip install TTS")

            # Set environment variable to auto-accept license
            os.environ["COQUI_TOS_AGREED"] = "1"

            # Temporarily monkey patch torch.load to disable weights_only for XTTS loading
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                # Initialize XTTS model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_path = f"tts_models/multilingual/multi-dataset/xtts_v2"
                tts = TTS(model_path).to(device)
            finally:
                # Restore original torch.load
                torch.load = original_load

            loaded_models[model_name] = {
                "type": "xtts",
                "tts": tts,
                "repo_id": repo_id
            }

        else:
            # Generic TTS pipeline
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline(
                "text-to-speech",
                model=repo_id,
                device=device,
            )

            loaded_models[model_name] = {
                "type": "generic_tts",
                "pipeline": pipe,
                "repo_id": repo_id
            }

        logger.info(f"Successfully loaded TTS model: {model_name}")
        return loaded_models[model_name]

    except Exception as e:
        logger.error(f"Failed to load TTS model {model_name}: {e}")
        raise


# --------------------------------------------------------------------------
# Transcription helpers (engine-agnostic dispatch + response building)
# --------------------------------------------------------------------------

def _resample_to_16k_mono(temp_path: str, background_tasks: BackgroundTasks) -> str:
    """Resample audio to 16kHz mono via ffmpeg if it isn't already.

    Returns the path to use for transcription. The original temp file is
    queued for background deletion if a new file was produced.
    """
    import wave
    import subprocess

    try:
        with wave.open(temp_path, "rb") as wf:
            sr = wf.getframerate()
    except wave.Error:
        # Not a parseable WAV — let the caller try anyway.
        return temp_path

    if sr == 16000:
        return temp_path

    resampled = temp_path + ".16k.wav"
    subprocess.run(
        ["ffmpeg", "-i", temp_path, "-ar", "16000", "-ac", "1", "-y", resampled],
        capture_output=True,
        timeout=30,
    )
    if os.path.exists(resampled):
        background_tasks.add_task(os.unlink, temp_path)
        return resampled
    return temp_path


def _run_stt(
    model_data: Dict[str, Any],
    temp_path: str,
    *,
    language: Optional[str],
    enable_vad: bool,
    chunk_length_s: Optional[int],
    want_segments: bool,
    background_tasks: BackgroundTasks,
):
    """Dispatch transcription to the right engine.

    Returns ``(text, detected_language, segments_or_None, duration_or_None)``.
    ``segments`` is a list of :class:`FormatSegment` ready for response
    formatting, or ``None`` when no timing data is available.
    """
    mtype = model_data["type"]
    norm_lang = language if language and language != "auto" else None

    # ---- Strategy-based engines (TranscriptionResult contract) ------------
    if mtype in ("moonshine", "sensevoice"):
        strategy = model_data["strategy"]
        r = strategy.transcribe(temp_path, language=norm_lang)
        segs = _segments_from_result(r) if want_segments else None
        return r.text, r.language, segs, getattr(r, "duration", None)

    if mtype == "faster-whisper":
        strategy = model_data["strategy"]
        kwargs: Dict[str, Any] = {"enable_vad": enable_vad}
        if chunk_length_s is not None:
            kwargs["chunk_length"] = chunk_length_s
        r = strategy.transcribe(temp_path, language=norm_lang, **kwargs)
        segs = _segments_from_result(r) if want_segments else None
        return r.text, r.language, segs, getattr(r, "duration", None)

    # ---- whisper-cpp (pywhispercpp) --------------------------------------
    if mtype == "whisper-cpp":
        path = _resample_to_16k_mono(temp_path, background_tasks)
        model = model_data["model"]
        segments = model.transcribe(path, language=norm_lang)
        text_parts = []
        fmt_segs: List[FormatSegment] = []
        for seg in segments:
            seg_text = getattr(seg, "text", "") or ""
            text_parts.append(seg_text.strip())
            if want_segments:
                # pywhispercpp t0/t1 are centiseconds (1/100s).
                start = getattr(seg, "t0", 0) / 100.0
                end = getattr(seg, "t1", 0) / 100.0
                fmt_segs.append(FormatSegment(start=start, end=end, text=seg_text))
        text = " ".join(p for p in text_parts if p)
        # Replace the path that the caller will background-delete.
        if path != temp_path:
            return text, language, (fmt_segs if want_segments else None), None
        return text, language, (fmt_segs if want_segments else None), None

    # ---- OpenAI Whisper (native) -----------------------------------------
    if mtype == "openai-whisper":
        model = model_data["model"]
        r = model.transcribe(temp_path, language=norm_lang)
        text = r["text"]
        detected = r.get("language")
        segs: Optional[List[FormatSegment]] = None
        if want_segments:
            segs = []
            for seg in r.get("segments") or []:
                segs.append(
                    FormatSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=str(seg.get("text", "")),
                    )
                )
            if not segs:
                segs = None
        return text, detected, segs, None

    # ---- HuggingFace transformers pipeline -------------------------------
    pipe = model_data["pipeline"]
    pipe_kwargs: Dict[str, Any] = {}
    if language:
        pipe_kwargs["generate_kwargs"] = {"language": language}
    if chunk_length_s is not None:
        pipe_kwargs["chunk_length_s"] = chunk_length_s
    pipe_kwargs["return_timestamps"] = want_segments
    result = pipe(temp_path, **pipe_kwargs)

    text = result["text"]
    detected = result.get("language")
    segs = None
    if want_segments and isinstance(result, dict):
        segs = []
        for chunk in result.get("chunks") or []:
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0]) if ts and ts[0] is not None else 0.0
            end = float(ts[1]) if ts and ts[1] is not None else start
            segs.append(FormatSegment(start=start, end=end, text=str(chunk.get("text", ""))))
        if not segs:
            segs = None
    return text, detected, segs, None


def _segments_from_result(r) -> Optional[List[FormatSegment]]:
    """Convert a strategy ``TranscriptionResult.segments`` to FormatSegments."""
    raw = getattr(r, "segments", None)
    if not raw:
        return None
    return [
        FormatSegment(start=float(s.start), end=float(s.end), text=str(s.text))
        for s in raw
    ]


def _build_transcription_response(
    *,
    text: str,
    language: Optional[str],
    segments: Optional[List[FormatSegment]],
    duration: Optional[float],
    model_name: str,
    response_format: str,
    include_timestamps: bool,
):
    """Build the HTTP response in the requested format."""
    if response_format == "text":
        return PlainTextResponse(text.strip() + "\n")

    if response_format == "markdown":
        body = to_markdown(
            text,
            segments,
            language=language,
            duration=duration,
            model=model_name,
        )
        return PlainTextResponse(body, media_type="text/markdown; charset=utf-8")

    if response_format == "srt":
        if not segments:
            raise HTTPException(
                status_code=422,
                detail="response_format='srt' requires segment timestamps, "
                "but the engine did not return any. Try a Whisper-family "
                "model (whisper, faster-whisper, whisper-cpp).",
            )
        return PlainTextResponse(to_srt(segments), media_type="application/x-subrip")

    if response_format == "vtt":
        if not segments:
            raise HTTPException(
                status_code=422,
                detail="response_format='vtt' requires segment timestamps, "
                "but the engine did not return any.",
            )
        return PlainTextResponse(to_vtt(segments), media_type="text/vtt; charset=utf-8")

    # Default JSON. Keep the v2.0.x shape for back-compat unless the caller
    # explicitly opts in to timestamps.
    body: Dict[str, Any] = {"text": text, "language": language}
    if duration is not None:
        body["duration"] = duration
    if include_timestamps and segments:
        body["segments"] = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments
        ]
    return JSONResponse(body)


def create_app(model_name: str) -> FastAPI:
    """Create FastAPI application for the specified model."""
    app = FastAPI(
        title=f"LocalKin Service Audio - {model_name} API",
        description=f"API server for {model_name} model",
        version="1.0.0"
    )

    model_info = _find_model_dict(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found")

    model_type = model_info.get("type")

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "LocalKin Service Audio API Server",
            "model": model_name,
            "type": model_type,
            "status": "running",
            "endpoints": {
                "GET /": "This information",
                "GET /health": "Health check",
                "GET /models": "Loaded models info",
                "POST /transcribe": "Speech to text (STT models)",
                "POST /synthesize": "Text to speech (TTS models)",
                "POST /chat": "Conversational interface (LLM models)"
            }
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": model_name,
            "loaded": model_name in loaded_models
        }

    @app.get("/models")
    async def get_models():
        """Get information about loaded models."""
        return {
            "loaded_models": list(loaded_models.keys()),
            "current_model": model_name,
            "model_info": model_info
        }

    if model_type == "stt":
        @app.post("/transcribe")
        async def transcribe_audio(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: Optional[str] = None,
            task: str = "transcribe",
            enable_vad: bool = True,
            timestamps: bool = False,
            response_format: str = "json",
            chunk_length_s: Optional[int] = None,
        ):
            """Transcribe audio to text.

            Query parameters:
              - language: BCP-47 code (e.g. ``en``, ``zh``). Default: auto-detect.
              - enable_vad: Apply Voice Activity Detection to skip silence.
                Currently honored by faster-whisper; ignored by engines that
                don't expose VAD. Default: True.
              - timestamps: Include segment-level timestamps in JSON output.
                Default: False (back-compat: original ``{"text", "language"}``
                shape is preserved).
              - response_format: ``json`` (default), ``text``, ``markdown``,
                ``srt``, or ``vtt``. Non-JSON formats automatically include
                segment timestamps when the engine produces them.
              - chunk_length_s: Override chunk length for VRAM tuning. Honored
                by faster-whisper and the HuggingFace pipeline; ignored
                elsewhere.
            """
            if response_format not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"response_format must be one of "
                        f"{sorted(SUPPORTED_FORMATS)}, got {response_format!r}"
                    ),
                )
            want_segments = timestamps or response_format != "json"

            try:
                if model_name not in loaded_models:
                    load_whisper_model(model_name)

                model_data = loaded_models[model_name]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name

                try:
                    text, detected_lang, fmt_segments, duration = _run_stt(
                        model_data,
                        temp_path,
                        language=language,
                        enable_vad=enable_vad,
                        chunk_length_s=chunk_length_s,
                        want_segments=want_segments,
                        background_tasks=background_tasks,
                    )
                    # Refresh path in case _run_stt resampled it (whisper-cpp).
                    background_tasks.add_task(os.unlink, temp_path)

                    return _build_transcription_response(
                        text=text,
                        language=language or detected_lang,
                        segments=fmt_segments,
                        duration=duration,
                        model_name=model_name,
                        response_format=response_format,
                        include_timestamps=timestamps,
                    )
                except HTTPException:
                    background_tasks.add_task(os.unlink, temp_path)
                    raise
                except Exception as e:
                    background_tasks.add_task(os.unlink, temp_path)
                    raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

        # OpenAI-compatible STT endpoint for backward compatibility with kin_listen
        @app.post("/v1/audio/transcriptions")
        async def openai_stt_compat(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: Optional[str] = None,
            model: Optional[str] = None,
            enable_vad: bool = True,
            timestamps: bool = False,
            response_format: str = "json",
            chunk_length_s: Optional[int] = None,
        ):
            """OpenAI-compatible STT endpoint. Maps to /transcribe."""
            return await transcribe_audio(
                background_tasks,
                file,
                language=language,
                enable_vad=enable_vad,
                timestamps=timestamps,
                response_format=response_format,
                chunk_length_s=chunk_length_s,
            )

    elif model_type == "tts":
        @app.post("/synthesize", response_model=TTSResponse)
        async def synthesize_speech(
            background_tasks: BackgroundTasks,
            request: TTSRequest
        ):
            """Synthesize text to speech."""
            try:
                # Load model if not loaded
                if model_name not in loaded_models:
                    load_tts_model(model_name)

                model_data = loaded_models[model_name]

                # Create output file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    output_path = temp_file.name

                try:
                    if model_data["type"] == "speecht5":
                        # SpeechT5 specific implementation
                        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                        import torch
                        import torchaudio

                        processor = model_data["processor"]
                        model = model_data["model"]
                        vocoder = model_data["vocoder"]

                        # Use default speaker embeddings (512-dimensional for SpeechT5 speaker embeddings)
                        # Create a neutral speaker embedding with some random variation
                        speaker_embeddings = torch.randn(1, 512, dtype=torch.float32) * 0.1

                        inputs = processor(text=request.text, return_tensors="pt")

                        # Generate speech
                        with torch.no_grad():
                            speech = model.generate_speech(
                                inputs["input_ids"],
                                speaker_embeddings=speaker_embeddings,
                                vocoder=vocoder
                            )

                        # Save to file
                        torchaudio.save(output_path, speech, 16000)

                        # Read the audio file and return it directly
                        with open(output_path, 'rb') as f:
                            audio_data = f.read()

                        # Clean up the temp file
                        os.unlink(output_path)

                        # Return audio file directly
                        return Response(
                            content=audio_data,
                            media_type="audio/wav",
                            headers={"Content-Disposition": "attachment; filename=speech.wav"}
                        )

                    elif model_data["type"] == "kokoro":
                        # Kokoro specific implementation
                        from kokoro import KPipeline
                        import soundfile as sf
                        import numpy as np

                        voice = request.speaker or 'af_heart'  # Default voice

                        # Select pipeline by voice language prefix (z=Chinese, j=Japanese, a=English, etc.)
                        lang_code = voice[0] if voice else 'a'
                        pipelines = model_data["pipelines"]
                        if lang_code not in pipelines:
                            pipelines[lang_code] = KPipeline(lang_code=lang_code)
                        pipeline = pipelines[lang_code]

                        # Generate speech using Kokoro
                        generator = pipeline(
                            request.text,
                            voice=voice,
                            speed=1.0,
                        )

                        # Collect all audio segments
                        audio_segments = []
                        for gs, ps, audio in generator:
                            audio_segments.append(audio)

                        # Concatenate all audio segments
                        if audio_segments:
                            final_audio = np.concatenate(audio_segments)
                        else:
                            final_audio = np.array([])

                        # Save to WAV file
                        sf.write(output_path, final_audio, 24000)  # Kokoro uses 24kHz

                        # Read the audio file and return it directly
                        with open(output_path, 'rb') as f:
                            audio_data = f.read()

                        # Clean up the temp file
                        os.unlink(output_path)

                        # Return audio file directly
                        return Response(
                            content=audio_data,
                            media_type="audio/wav",
                            headers={"Content-Disposition": "attachment; filename=speech.wav"}
                        )

                    elif model_data["type"] == "xtts":
                        # XTTS specific implementation
                        tts = model_data["tts"]

                        # Generate speech
                        output_path_temp = tempfile.mktemp(suffix=".wav")
                        # Use default speaker and language for XTTS v2
                        # XTTS v2 uses specific speaker names - try different ones
                        try:
                            # Try with a common XTTS speaker name
                            tts.tts_to_file(
                                text=request.text,
                                file_path=output_path_temp,
                                speaker="Claribel Dervla",  # Known XTTS speaker
                                language="en"
                            )
                        except Exception as e:
                            if "speaker" in str(e).lower():
                                # If speaker fails, try without speaker (may use default)
                                try:
                                    tts.tts_to_file(
                                        text=request.text,
                                        file_path=output_path_temp,
                                        language="en"
                                    )
                                except Exception as e2:
                                    # Last resort - check available speakers
                                    try:
                                        speakers = getattr(tts, 'speakers', None) or getattr(tts.tts_model, 'speakers', None)
                                        if speakers:
                                            speaker_name = list(speakers.keys())[0] if speakers else "en_0"
                                        else:
                                            speaker_name = "en_0"
                                        tts.tts_to_file(
                                            text=request.text,
                                            file_path=output_path_temp,
                                            speaker=speaker_name,
                                            language="en"
                                        )
                                    except Exception as e3:
                                        raise RuntimeError(f"All XTTS speaker combinations failed. Last error: {str(e3)}")
                            else:
                                raise e

                        # Read the audio file and return it directly
                        with open(output_path_temp, 'rb') as f:
                            audio_data = f.read()

                        # Clean up the temp file
                        os.unlink(output_path_temp)

                        # Return audio file directly
                        return Response(
                            content=audio_data,
                            media_type="audio/wav",
                            headers={"Content-Disposition": "attachment; filename=speech.wav"}
                        )

                    elif model_data["type"] == "bark":
                        # Bark specific implementation
                        from transformers import BarkProcessor, BarkModel
                        import torch
                        import scipy

                        processor = model_data["processor"]
                        model = model_data["model"]

                        # Bark uses special voice presets (e.g., "v2/en_speaker_0" through "v2/en_speaker_9")
                        voice_preset = request.speaker or "v2/en_speaker_6"  # Default to speaker 6

                        # Process inputs
                        inputs = processor(request.text, voice_preset=voice_preset, return_tensors="pt")

                        # Move to device if available
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = model.to(device)
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        # Generate audio
                        with torch.no_grad():
                            audio_array = model.generate(**inputs)

                        # Convert to numpy and squeeze
                        audio_array = audio_array.cpu().numpy().squeeze()

                        # Bark uses 24kHz sample rate
                        sample_rate = model.generation_config.sample_rate

                        # Save to file using scipy
                        scipy.io.wavfile.write(output_path, rate=sample_rate, data=audio_array)

                        # Read the audio file and return it directly
                        with open(output_path, 'rb') as f:
                            audio_data = f.read()

                        # Clean up the temp file
                        os.unlink(output_path)

                        # Return audio file directly
                        return Response(
                            content=audio_data,
                            media_type="audio/wav",
                            headers={"Content-Disposition": "attachment; filename=speech.wav"}
                        )

                    else:
                        # Generic TTS pipeline
                        pipe = model_data["pipeline"]
                        result = pipe(request.text)

                        # Save result (implementation depends on specific model)
                        # This is a placeholder for generic TTS handling
                        raise HTTPException(
                            status_code=501,
                            detail=f"TTS implementation for {model_data['type']} not yet implemented"
                        )

                except Exception as e:
                    # Clean up temp file on error
                    if os.path.exists(output_path):
                        background_tasks.add_task(os.unlink, output_path)
                    raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    # OpenAI-compatible TTS endpoint for backward compatibility with kin_speak
    if model_type == "tts":
        @app.post("/v1/audio/speech")
        async def openai_tts_compat(
            background_tasks: BackgroundTasks,
            request: Dict[str, Any],
        ):
            """OpenAI-compatible TTS endpoint. Maps to /synthesize."""
            tts_req = TTSRequest(
                text=request.get("input", ""),
                speaker=request.get("voice"),
            )
            return await synthesize_speech(background_tasks, tts_req)

    @app.post("/chat")
    async def chat(request: Dict[str, Any]):
        """Chat endpoint for conversational models (future implementation)."""
        return {
            "message": "Chat functionality not yet implemented for this model type",
            "model": model_name,
            "type": model_type
        }

    # Include UI routes if available
    try:
        ui_router = create_ui_router()
        app.include_router(ui_router, prefix="", tags=["ui"])

        # Mount static files for UI
        ui_static_path = Path(__file__).parent.parent / "ui" / "static"
        if ui_static_path.exists():
            app.mount("/ui/static", StaticFiles(directory=str(ui_static_path)), name="ui-static")

        logger.info("🌐 Web UI routes enabled")
    except ImportError:
        logger.info("ℹ️  Web UI not available (ui module not found)")

    return app

def run_server(model_name: str, host: str = "0.0.0.0", port: int = 8000):
    """Run the API server for the specified model."""
    try:
        logger.info(f"🚀 Starting LocalKin Service Audio API server for {model_name}")
        logger.info(f"📍 Server will be available at: http://{host}:{port}")
        logger.info(f"📖 API documentation: http://{host}:{port}/docs")

        app = create_app(model_name)

        # Show available endpoints
        logger.info("🔧 Available API endpoints:")
        logger.info(f"   GET  /           - API information")
        logger.info(f"   GET  /health     - Health check")
        logger.info(f"   GET  /models     - Loaded models info")
        logger.info(f"   GET  /docs       - Interactive API documentation")

        model_info = _find_model_dict(model_name)
        if model_info:
            model_type = model_info.get("type")
            if model_type == "stt":
                logger.info(f"   POST /transcribe - Speech to text")
            elif model_type == "tts":
                logger.info(f"   POST /synthesize - Text to speech")
            else:
                logger.info(f"   POST /chat       - Chat interface")

        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
