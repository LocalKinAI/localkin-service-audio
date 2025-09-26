"""
Web UI routes for LocalKin Service Audio.

This module contains FastAPI routes for the web-based user interface,
providing a modern, interactive way to use LocalKin Service Audio's audio processing capabilities.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.responses import Response as StarletteResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core import (
    get_models, find_model, list_local_models,
    transcribe_audio, synthesize_speech, get_cache_info
)
from ..cli.cli import synthesize_huggingface_tts, print_error, print_info
from ..templates import list_available_templates


def discover_api_servers() -> dict:
    """Discover running API servers by scanning common ports."""
    import requests
    discovered_servers = {}

    # Common ports where API servers might be running
    common_ports = [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010]

    print("🔍 Scanning for running API servers...")
    for port in common_ports:
        try:
            health_url = f"http://localhost:{port}/health"
            response = requests.get(health_url, timeout=0.5)
            if response.status_code == 200:
                health_data = response.json()
                model = health_data.get('model', 'unknown')
                is_loaded = health_data.get('loaded', False)

                if is_loaded:
                    discovered_servers[model] = port
                    print(f"✅ Found {model} API server on port {port} (ready)")
                else:
                    print(f"⏳ Found {model} API server on port {port} (loading...)")
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            continue  # Port not responding, skip

    return discovered_servers


def synthesize_via_api_to_file(text: str, model_name: str, output_path: str) -> bool:
    """Try to synthesize speech via API server and save to file."""
    try:
        # Discover running API servers dynamically
        api_servers = discover_api_servers()

        # Check if requested model has a running API server
        if model_name not in api_servers:
            print(f"🔍 No running API server found for {model_name}, will use local synthesis")
            return False

        port = api_servers[model_name]

        # API server is available, send synthesis request
        import requests
        api_url = f"http://localhost:{port}/synthesize"
        payload = {"text": text}

        print(f"🌐 Using discovered API server at localhost:{port} for {model_name}")

        # Use longer timeout for complex models when API server is confirmed loaded
        timeout = 60 if model_name in ['kokoro-82m', 'xtts-v2', 'speecht5-tts'] else 30
        print(f"⏳ Using {timeout}s timeout for {model_name} synthesis")

        response = requests.post(api_url, json=payload, timeout=timeout)

        if response.status_code == 200:
            # Save the audio data to the output file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ TTS synthesized via API server (localhost:{port})")
            return True
        else:
            print(f"❌ API server returned error: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ API call failed for {model_name}: {e}")
        return False


def synthesize_via_api_to_file_with_params(params: dict, model_name: str, output_path: str) -> bool:
    """Try to synthesize speech via API server with parameters and save to file."""
    try:
        # Discover running API servers dynamically
        api_servers = discover_api_servers()

        # Check if requested model has a running API server
        if model_name not in api_servers:
            print(f"🔍 No running API server found for {model_name}, will use local synthesis")
            return False

        port = api_servers[model_name]

        # API server is available, send synthesis request
        import requests
        api_url = f"http://localhost:{port}/synthesize"
        payload = params

        print(f"🌐 Using discovered API server at localhost:{port} for {model_name}")

        # Use longer timeout for complex models when API server is confirmed loaded
        timeout = 60 if model_name in ['kokoro-82m', 'xtts-v2', 'speecht5-tts'] else 30
        print(f"⏳ Using {timeout}s timeout for {model_name} synthesis")

        response = requests.post(api_url, json=payload, timeout=timeout)

        if response.status_code == 200:
            # Save the audio data to the output file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ TTS synthesized via API server (localhost:{port})")
            return True
        else:
            print(f"❌ API server returned error: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ API call failed for {model_name}: {e}")
        return False


def synthesize_huggingface_tts_with_voice(model_name: str, text: str, output_path: str = None, voice: str = None) -> bool:
    """Synthesize speech using Hugging Face TTS models with voice selection."""
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

        # Handle different TTS model types with voice support
        if "kokoro" in model_name.lower():
            # Kokoro TTS implementation with voice
            print_info("🔧 Initializing Kokoro TTS...")
            try:
                # Ensure spaCy model is available (same setup as API server)
                import os
                import sys
                system_site_packages = '/Users/jackysun/.pyenv/versions/3.10.0/lib/python3.10/site-packages'
                if system_site_packages not in sys.path:
                    sys.path.insert(0, system_site_packages)

                from kokoro import KPipeline
                import torch

                # Map voice names to kokoro voices
                voice_mapping = {
                    'male': 'am_adam',
                    'female': 'af_sarah',
                    'af_sarah': 'af_sarah',
                    'am_adam': 'am_adam',
                    'default': 'af_sarah'
                }
                selected_voice = voice_mapping.get(voice or 'default', 'af_sarah')

                try:
                    pipeline = KPipeline(lang_code='a')  # American English
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
                generator = pipeline(text, voice=selected_voice, speed=1, split_pattern=r'\n+')

                # Generate audio
                audio_data = []
                for i, (gs, ps, audio) in enumerate(generator):
                    audio_data.append(audio)

                if audio_data:
                    # Concatenate audio segments
                    import numpy as np
                    combined_audio = np.concatenate(audio_data)

                    # Save to file
                    import soundfile as sf
                    sf.write(output_path, combined_audio, 24000)
                    print_info(f"✅ Kokoro TTS synthesis complete with voice: {selected_voice}")
                    return True
                else:
                    print_error("No audio generated by Kokoro")
                    return False

            except Exception as e:
                if "spacy" in str(e).lower() or "en_core_web_sm" in str(e).lower():
                    raise RuntimeError(
                        "Kokoro TTS requires spaCy English model. "
                        "Install it with: python -m spacy download en_core_web_sm"
                    )
                else:
                    print_error(f"Kokoro TTS failed: {e}")
                    return False

        elif "xtts" in model_name.lower():
            # XTTS v2 with speaker selection
            try:
                from TTS.api import TTS
                import os

                # Set environment variable to auto-accept license
                os.environ["COQUI_TOS_AGREED"] = "1"

                # Temporarily monkey patch torch.load to disable weights_only for XTTS loading
                import torch
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

                    # Use default speaker if none specified
                    speaker = voice or "Claribel Dervla"
                    print_info(f"🎭 Using XTTS speaker: {speaker}")

                    # Generate speech
                    tts.tts_to_file(
                        text=text,
                        file_path=output_path,
                        speaker=speaker,
                        language="en"
                    )

                    print_info("✅ XTTS v2 synthesis complete")
                    return True

                finally:
                    # Restore original torch.load
                    torch.load = original_load

            except Exception as e:
                print_error(f"XTTS synthesis failed: {e}")
                return False

        else:
            # Fallback to regular synthesis without voice
            from ..cli.cli import synthesize_huggingface_tts
            return synthesize_huggingface_tts(model_name, text, output_path)

    except Exception as e:
        print_error(f"HuggingFace TTS with voice failed: {e}")
        return False


# Router for UI endpoints
router = APIRouter()

# Templates directory
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Temporary directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "localkin_service_audio_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Output directory for generated files
OUTPUT_DIR = Path(tempfile.gettempdir()) / "localkin_service_audio_output"
OUTPUT_DIR.mkdir(exist_ok=True)


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface page."""
    models = get_models()
    cache_info = get_cache_info()
    templates_list = list_available_templates()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "cache_info": cache_info,
        "templates": templates_list,
        "current_year": datetime.now().year
    })


@router.get("/transcribe", response_class=HTMLResponse)
async def transcribe_page(request: Request):
    """Speech-to-text interface page."""
    models = [m for m in get_models() if m.get("type") == "stt"]
    return templates.TemplateResponse("transcribe.html", {
        "request": request,
        "models": models,
        "current_year": datetime.now().year
    })


@router.get("/synthesize", response_class=HTMLResponse)
async def synthesize_page(request: Request):
    """Text-to-speech interface page."""
    models = [m for m in get_models() if m.get("type") == "tts"]
    return templates.TemplateResponse("synthesize.html", {
        "request": request,
        "models": models,
        "current_year": datetime.now().year
    })


@router.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("whisper-tiny-hf")
):
    """API endpoint for transcription via web interface."""
    try:
        # Validate model
        model = find_model(model_name)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

        # Save uploaded file
        file_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Determine model and engine (similar to CLI logic)
        if model_name.startswith("faster-whisper"):
            # Extract model size for faster-whisper models
            model_size = model_name.replace("faster-whisper-", "")
            engine = "faster"
        elif model_name == "whisper":
            # Default whisper model - use base as default size
            model_size = "base"
            engine = "openai"
        elif model_name.startswith("whisper-"):
            # Specific whisper model size
            model_size = model_name.replace("whisper-", "")
            engine = "openai"
        else:
            # For other models, use auto detection
            model_size = model_name
            engine = "auto"

        # Transcribe audio
        transcription = transcribe_audio(model_size, str(file_path), engine)

        # Clean up uploaded file
        file_path.unlink(missing_ok=True)

        # Check if transcription was successful
        if transcription.startswith("Error:"):
            raise HTTPException(status_code=500, detail=transcription)

        return {
            "success": True,
            "text": transcription,
            "language": "unknown",  # transcribe_audio doesn't return language info
            "confidence": 0.0,      # transcribe_audio doesn't return confidence
            "model": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(None),
    file: UploadFile = File(None),
    model_name: str = Form("speecht5-tts"),
    voice: str = Form(None)
):
    """API endpoint for speech synthesis via web interface."""
    try:
        # Validate model
        model = find_model(model_name)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

        # Get text content - either from form or file
        if file and file.filename:
            # Read text from uploaded file
            content = await file.read()
            text_content = content.decode('utf-8')
            print(f"📄 Read {len(text_content)} characters from file: {file.filename}")
        elif text:
            text_content = text
        else:
            raise HTTPException(status_code=400, detail="Either text or file must be provided")

        # Validate text length
        if len(text_content) > 10000:  # Increased limit for file input
            raise HTTPException(status_code=400, detail="Text too long (max 10000 characters)")

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"synthesis_{timestamp}.wav"
        output_path = OUTPUT_DIR / output_filename

        # Prepare synthesis parameters
        synthesis_text = text_content
        synthesis_params = {"text": synthesis_text}

        # Add voice parameter for supported models
        if voice and model_name in ['kokoro-82m', 'xtts-v2', 'speecht5-tts', 'tortoise-tts']:
            synthesis_params["voice"] = voice
            print(f"🎭 Using voice: {voice} for {model_name}")

        # Try API server first for better performance with complex models
        api_success = synthesize_via_api_to_file_with_params(synthesis_params, model_name, str(output_path))

        if not api_success:
            # Check if this is native TTS (allowed to fallback locally)
            if model_name == 'native':
                # Use native TTS locally
                result = synthesize_speech(synthesis_text, str(output_path))
                if not output_path.exists():
                    raise HTTPException(status_code=500, detail="Speech synthesis failed")
            else:
                # For non-native models, report API unavailability
                import json
                error_details = {
                    "success": False,
                    "error": "API_UNAVAILABLE",
                    "message": f"API server not available for {model_name}",
                    "solution": f"Start the API server with: kin audio run {model_name} --port 8001",
                    "model": model_name
                }
                # Return API unavailability as 200 with error details in JSON
                error_details["http_status"] = 503
                return StarletteResponse(
                    content=json.dumps(error_details),
                    media_type="application/json",
                    status_code=200
                )

        success_response = {
            "success": True,
            "audio_url": f"/ui/audio/{output_filename}",
            "model": model_name,
            "voice": voice or "default",
            "text_length": len(text_content),
            "file_size": output_path.stat().st_size,
            "input_type": "file" if file and file.filename else "text"
        }
        return StarletteResponse(
            content=json.dumps(success_response),
            media_type="application/json",
            status_code=200
        )

    except Exception as e:
        # For debugging, return error as JSON instead of raising HTTPException
        error_response = {
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": str(e),
            "model": model_name
        }
        return StarletteResponse(
            content=json.dumps(error_response),
            media_type="application/json",
            status_code=500
        )


@router.get("/api/models")
async def api_get_models():
    """Get available models for the web interface."""
    models = get_models()
    cache_info = get_cache_info()

    return {
        "models": models,
        "cache_info": cache_info,
        "stt_models": [m for m in models if m.get("type") == "stt"],
        "tts_models": [m for m in models if m.get("type") == "tts"]
    }


@router.get("/api/status")
async def api_status():
    """Get system status for the web interface."""
    cache_info = get_cache_info()
    models = get_models()

    return {
        "status": "running",
        "models_count": len(models),
        "cached_models": len(cache_info.get("cached_models", [])),
        "total_cache_size": sum(m.get("size_mb", 0) for m in cache_info.get("cached_models", [])),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/ui/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )


def create_ui_router() -> APIRouter:
    """Create and return the UI router with all routes configured."""
    return router
