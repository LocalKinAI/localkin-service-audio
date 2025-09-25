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
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core import (
    get_models, find_model, list_local_models,
    transcribe_audio, synthesize_speech, get_cache_info
)
from ..templates import list_available_templates

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

        # Transcribe audio
        result = transcribe_audio(str(file_path))

        # Clean up uploaded file
        file_path.unlink(missing_ok=True)

        return {
            "success": True,
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "model": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/synthesize")
async def api_synthesize(
    text: str = Form(...),
    model_name: str = Form("speecht5-tts")
):
    """API endpoint for speech synthesis via web interface."""
    try:
        # Validate model
        model = find_model(model_name)
        if not model:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not found")

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"synthesis_{timestamp}.wav"
        output_path = OUTPUT_DIR / output_filename

        # Synthesize speech
        result = synthesize_speech(text, str(output_path))

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Speech synthesis failed")

        return {
            "success": True,
            "audio_url": f"/ui/audio/{output_filename}",
            "model": model_name,
            "text_length": len(text),
            "file_size": output_path.stat().st_size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/audio/{filename}")
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
