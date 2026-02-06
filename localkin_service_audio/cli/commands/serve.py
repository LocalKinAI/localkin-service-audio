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
def serve(
    model_name: Optional[str],
    host: str,
    port: int,
    reload: bool
):
    """
    Start the API server.

    Examples:

        kin audio serve

        kin audio serve --port 8001

        kin audio serve kokoro --port 8001
    """
    print_header("API Server")
    print_info(f"Starting server on http://{host}:{port}")

    if model_name:
        print_info(f"Pre-loading model: {model_name}")

    try:
        import uvicorn
        from ...api.server import app

        # Pre-load model if specified
        if model_name:
            from ...core import get_audio_engine
            engine = get_audio_engine()

            # Determine if STT or TTS based on model
            from ...core.config import model_registry
            model_config = model_registry.get(model_name)

            if model_config:
                if model_config.type.value == "stt":
                    engine.load_stt(model_name)
                else:
                    engine.load_tts(model_name)
                print_success(f"Loaded {model_name}")

        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
        )

    except ImportError:
        print_error("uvicorn not installed. Install with: pip install uvicorn")
        raise click.Abort()
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
