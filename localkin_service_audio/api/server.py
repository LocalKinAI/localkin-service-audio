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
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from ..core import find_model
from ..ui import create_ui_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration (duplicated from models.py to avoid circular imports)
from pathlib import Path
HF_CACHE_DIR = Path.home() / ".localkin-service-audio" / "cache" / "huggingface"
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

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    language: Optional[str] = None

class TTSResponse(BaseModel):
    audio_path: str
    duration: Optional[float] = None

def load_whisper_model(model_name: str):
    """Load a Whisper model from Hugging Face."""
    try:
        from transformers import pipeline
        import torch

        model_info = find_model(model_name)
        if not model_info or model_info.get("source") != "huggingface":
            raise ValueError(f"Model {model_name} not found or not a Hugging Face model")

        repo_id = model_info.get("huggingface_repo")
        if not repo_id:
            raise ValueError(f"No Hugging Face repo specified for {model_name}")

        logger.info(f"Loading Whisper model: {repo_id}")

        # Load the model pipeline
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
    """Load a TTS model from Hugging Face."""
    try:
        from transformers import pipeline
        import torch

        model_info = find_model(model_name)
        if not model_info or model_info.get("source") != "huggingface":
            raise ValueError(f"Model {model_name} not found or not a Hugging Face model")

        repo_id = model_info.get("huggingface_repo")
        if not repo_id:
            raise ValueError(f"No Hugging Face repo specified for {model_name}")

        logger.info(f"Loading TTS model: {repo_id}")

        # Load TTS pipeline based on model type
        if "speecht5" in model_name.lower():
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
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
            # You can extend this to support other languages
            # Ensure spaCy model is available
            import os
            import sys
            system_site_packages = '/Users/jackysun/.pyenv/versions/3.10.0/lib/python3.10/site-packages'
            if system_site_packages not in sys.path:
                sys.path.insert(0, system_site_packages)

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
                "pipeline": pipeline,
                "repo_id": repo_id,
                "lang_code": 'a'  # Default language
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

def create_app(model_name: str) -> FastAPI:
    """Create FastAPI application for the specified model."""
    app = FastAPI(
        title=f"LocalKin Service Audio - {model_name} API",
        description=f"API server for {model_name} model",
        version="1.0.0"
    )

    model_info = find_model(model_name)
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
        @app.post("/transcribe", response_model=TranscriptionResponse)
        async def transcribe_audio(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: Optional[str] = None,
            task: str = "transcribe"
        ):
            """Transcribe audio file to text."""
            try:
                # Load model if not loaded
                if model_name not in loaded_models:
                    load_whisper_model(model_name)

                model_data = loaded_models[model_name]
                pipe = model_data["pipeline"]

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_path = temp_file.name

                try:
                    # Run transcription
                    result = pipe(
                        temp_path,
                        generate_kwargs={"language": language} if language else {},
                        return_timestamps=False
                    )

                    # Clean up temp file
                    background_tasks.add_task(os.unlink, temp_path)

                    return TranscriptionResponse(
                        text=result["text"],
                        language=language or result.get("language"),
                        confidence=result.get("confidence")
                    )

                except Exception as e:
                    # Clean up temp file on error
                    background_tasks.add_task(os.unlink, temp_path)
                    raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

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

                        pipeline = model_data["pipeline"]
                        voice = request.speaker or 'af_heart'  # Default voice

                        # Generate speech using Kokoro
                        generator = pipeline(
                            request.text,
                            voice=voice,
                            speed=1.0,  # You can add speed control to TTSRequest if needed
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

        model_info = find_model(model_name)
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
