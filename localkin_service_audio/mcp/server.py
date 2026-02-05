"""
MCP Server for LocalKin Audio.

Provides tools for Claude Code/Desktop to:
- Transcribe audio files
- Synthesize speech from text
- Clone voices
- List available models and voices

Usage:
    # Add to Claude Code settings (~/.claude/settings.json):
    {
        "mcpServers": {
            "localkin-audio": {
                "command": "kin",
                "args": ["mcp"]
            }
        }
    }

    # Or run directly:
    python -m localkin_service_audio.mcp.server
"""
import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None


def create_server() -> "Server":
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP not installed. Install with: pip install mcp"
        )

    server = Server("localkin-audio")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="transcribe_audio",
                description="Transcribe an audio file to text using speech-to-text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to transcribe"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use (e.g., whisper-cpp:base, sensevoice:small)",
                            "default": "whisper-cpp:base"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language code (e.g., en, zh). Auto-detect if not specified."
                        }
                    },
                    "required": ["audio_path"]
                }
            ),
            Tool(
                name="synthesize_speech",
                description="Convert text to speech audio file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to convert to speech"
                        },
                        "model": {
                            "type": "string",
                            "description": "TTS model to use (e.g., kokoro, cosyvoice:300m)",
                            "default": "kokoro"
                        },
                        "voice": {
                            "type": "string",
                            "description": "Voice ID to use"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the audio file. Auto-generated if not specified."
                        }
                    },
                    "required": ["text"]
                }
            ),
            Tool(
                name="clone_voice",
                description="Clone a voice from reference audio and synthesize new text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reference_audio": {
                            "type": "string",
                            "description": "Path to reference audio (3-10 seconds recommended)"
                        },
                        "text": {
                            "type": "string",
                            "description": "Text to synthesize with cloned voice"
                        },
                        "reference_text": {
                            "type": "string",
                            "description": "Transcript of reference audio (optional, improves quality)"
                        },
                        "model": {
                            "type": "string",
                            "description": "Voice cloning model (cosyvoice:300m, f5-tts)",
                            "default": "cosyvoice:300m"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to save the output audio"
                        }
                    },
                    "required": ["reference_audio", "text"]
                }
            ),
            Tool(
                name="list_models",
                description="List available STT and TTS models",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["all", "stt", "tts"],
                            "description": "Filter by model type",
                            "default": "all"
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by language support (e.g., zh, en)"
                        }
                    }
                }
            ),
            Tool(
                name="list_voices",
                description="List available voices for a TTS model",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "TTS model name (e.g., kokoro, cosyvoice:300m)",
                            "default": "kokoro"
                        }
                    }
                }
            ),
            Tool(
                name="get_audio_info",
                description="Get information about an audio file (duration, sample rate, etc.)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file"
                        }
                    },
                    "required": ["audio_path"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        """Handle tool calls."""
        try:
            if name == "transcribe_audio":
                result = await _transcribe_audio(
                    arguments["audio_path"],
                    arguments.get("model", "whisper-cpp:base"),
                    arguments.get("language"),
                )
            elif name == "synthesize_speech":
                result = await _synthesize_speech(
                    arguments["text"],
                    arguments.get("model", "kokoro"),
                    arguments.get("voice"),
                    arguments.get("output_path"),
                )
            elif name == "clone_voice":
                result = await _clone_voice(
                    arguments["reference_audio"],
                    arguments["text"],
                    arguments.get("reference_text"),
                    arguments.get("model", "cosyvoice:300m"),
                    arguments.get("output_path"),
                )
            elif name == "list_models":
                result = await _list_models(
                    arguments.get("type", "all"),
                    arguments.get("language"),
                )
            elif name == "list_voices":
                result = await _list_voices(
                    arguments.get("model", "kokoro"),
                )
            elif name == "get_audio_info":
                result = await _get_audio_info(
                    arguments["audio_path"],
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]

    return server


async def _transcribe_audio(
    audio_path: str,
    model: str = "whisper-cpp:base",
    language: Optional[str] = None,
) -> dict:
    """Transcribe audio file."""
    from ..core.audio_processing import get_audio_engine

    engine = get_audio_engine()

    # Load model if needed
    if not engine._stt_strategy or engine._stt_model_name != model:
        success = await asyncio.to_thread(engine.load_stt, model)
        if not success:
            return {"error": f"Failed to load model: {model}"}

    # Transcribe
    result = await asyncio.to_thread(
        engine.transcribe,
        audio_path,
        language=language,
    )

    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "model": model,
        "emotion": result.emotion,  # SenseVoice
    }


async def _synthesize_speech(
    text: str,
    model: str = "kokoro",
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Synthesize speech from text."""
    from ..core.audio_processing import get_audio_engine

    engine = get_audio_engine()

    # Load model if needed
    if not engine._tts_strategy or engine._tts_model_name != model:
        success = await asyncio.to_thread(engine.load_tts, model)
        if not success:
            return {"error": f"Failed to load model: {model}"}

    # Synthesize
    result = await asyncio.to_thread(
        engine.synthesize,
        text,
        voice=voice,
    )

    # Save to file
    if not output_path:
        output_path = str(Path(tempfile.gettempdir()) / f"tts_{uuid.uuid4().hex[:8]}.wav")

    await asyncio.to_thread(result.save, output_path)

    return {
        "output_path": output_path,
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "model": model,
        "voice": voice,
    }


async def _clone_voice(
    reference_audio: str,
    text: str,
    reference_text: Optional[str] = None,
    model: str = "cosyvoice:300m",
    output_path: Optional[str] = None,
) -> dict:
    """Clone voice and synthesize text."""
    from ..core.audio_processing import get_audio_engine

    engine = get_audio_engine()

    # Load model if needed
    if not engine._tts_strategy or engine._tts_model_name != model:
        success = await asyncio.to_thread(engine.load_tts, model)
        if not success:
            return {"error": f"Failed to load model: {model}"}

    # Clone voice
    result = await asyncio.to_thread(
        engine.clone_voice,
        reference_audio,
        text,
        reference_text,
    )

    # Save to file
    if not output_path:
        output_path = str(Path(tempfile.gettempdir()) / f"clone_{uuid.uuid4().hex[:8]}.wav")

    await asyncio.to_thread(result.save, output_path)

    return {
        "output_path": output_path,
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "model": model,
    }


async def _list_models(
    model_type: str = "all",
    language: Optional[str] = None,
) -> dict:
    """List available models."""
    from ..core.config import model_registry
    from ..core.types import ModelType

    if model_type == "stt":
        models = model_registry.list_stt_models()
    elif model_type == "tts":
        models = model_registry.list_tts_models()
    else:
        models = model_registry.list_all()

    if language:
        models = [m for m in models if language in m.languages]

    return {
        "models": [
            {
                "name": m.name,
                "type": m.type.value,
                "engine": m.engine,
                "languages": m.languages,
                "features": m.features,
                "description": m.description,
            }
            for m in models
        ],
        "total": len(models),
    }


async def _list_voices(model: str = "kokoro") -> dict:
    """List available voices for a model."""
    from ..core.audio_processing import get_audio_engine

    engine = get_audio_engine()

    # Load model if needed
    if not engine._tts_strategy or engine._tts_model_name != model:
        success = await asyncio.to_thread(engine.load_tts, model)
        if not success:
            return {"error": f"Failed to load model: {model}"}

    voices = engine.list_voices()

    return {
        "model": model,
        "voices": [
            {
                "id": v.id,
                "name": v.name,
                "language": v.language,
                "gender": v.gender,
            }
            for v in voices
        ],
        "total": len(voices),
    }


async def _get_audio_info(audio_path: str) -> dict:
    """Get audio file information."""
    import soundfile as sf
    import os

    if not os.path.exists(audio_path):
        return {"error": f"File not found: {audio_path}"}

    info = sf.info(audio_path)

    return {
        "path": audio_path,
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
    }


def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("MCP not installed. Install with: pip install mcp")
        return

    server = create_server()

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(main())


if __name__ == "__main__":
    run_server()
