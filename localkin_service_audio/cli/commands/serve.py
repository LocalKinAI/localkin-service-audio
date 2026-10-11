"""
Serve command - API server operations.
"""
import click
from typing import Optional

from ..utils import print_success, print_error, print_info, print_header


@click.command()
@click.argument("model_name", required=False)
@click.option(
    "--host", "-h",
    default="127.0.0.1",
    help="Host to bind to."
)
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="Port to bind to."
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development."
)
@click.option(
    "--emotion",
    "emotion_model",
    default=None,
    help="Sidecar model for emotion/audio events on /transcribe, e.g. sensevoice:small."
)
@click.option(
    "--preload/--no-preload",
    default=True,
    help="Load the model before accepting requests (default) or on the first request."
)
@click.option("--stt", "stt_model", default=None,
              help="Speech-to-text model; with --tts, both are served on one port.")
@click.option("--tts", "tts_model", default=None, help="Text-to-speech model; see --stt.")
def serve(
    model_name: Optional[str],
    host: str,
    port: int,
    reload: bool,
    emotion_model: Optional[str],
    preload: bool,
    stt_model: Optional[str],
    tts_model: Optional[str],
):
    """
    Start the API server.

    Examples:

        kin audio serve

        kin audio serve --port 8001

        kin audio serve kokoro --port 8001

        kin audio serve qwen3-asr:1.7b --emotion sensevoice:small

        # Both on one port: one base URL for OpenAI clients and agents
        kin audio serve --stt qwen3-asr:1.7b --tts qwen3-tts:0.6b
    """
    print_header("API Server")
    print_info(f"Starting server on http://{host}:{port}")

    if model_name and (stt_model or tts_model):
        print_error("Give a model name, or --stt/--tts, not both")
        raise click.Abort()
    names = [n for n in (model_name, stt_model, tts_model) if n]
    if names and preload:
        print_info(f"Pre-loading: {', '.join(names)}")

    try:
        import uvicorn
    except ImportError:
        print_error("uvicorn not installed. Install with: pip install uvicorn")
        raise click.Abort()

    try:
        from ...api.server import create_app, create_combined_app

        if not names:
            print_error("Model name required. Example: kin audio serve qwen3-tts:0.6b, "
                        "or kin audio serve --stt qwen3-asr:1.7b --tts qwen3-tts:0.6b")
            raise click.Abort()

        if model_name:
            app = create_app(model_name, emotion_model=emotion_model, preload=preload)
        else:
            app = create_combined_app(stt_model, tts_model, emotion_model=emotion_model, preload=preload)
        if preload:
            print_success(f"Loaded {', '.join(names)}")

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
        )

    except Exception as e:
        print_error(f"Server failed: {e}")
        raise click.Abort()


# Alias for backward compatibility
run = serve


@click.command()
@click.option(
    "--host", "-h",
    default="127.0.0.1",
    help="Host to bind to."
)
@click.option(
    "--port", "-p",
    type=int,
    default=8080,
    help="Port to bind to."
)
def web(host: str, port: int):
    """
    Start the web interface.

    Examples:

        kin web

        kin web --port 8080
    """
    print_header("Web Interface")
    print_info(f"Starting web UI on http://{host}:{port}")

    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from pathlib import Path
        from ...ui.routes import create_ui_router

        app = FastAPI(title="LocalKin Audio - Web UI")
        app.include_router(create_ui_router())

        # Mount static files if the directory exists
        static_dir = Path(__file__).parent.parent.parent / "ui" / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        uvicorn.run(app, host=host, port=port)

    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        raise click.Abort()
    except Exception as e:
        print_error(f"Web server failed: {e}")
        raise click.Abort()
