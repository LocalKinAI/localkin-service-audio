"""
CLI command modules.
"""
from .transcribe import transcribe
from .synthesize import synthesize, tts
from .models import models, pull, rm
from .serve import serve, run
from .recommend import recommend
from .benchmark import benchmark
from .listen import listen
from .config import config
from .status import status
from .cache import cache
from .ps import ps
from .add_model import add_model
from .list_templates import list_templates
from .music import music

__all__ = [
    "transcribe",
    "synthesize",
    "tts",
    "models",
    "pull",
    "rm",
    "serve",
    "run",
    "recommend",
    "benchmark",
    "listen",
    "config",
    "status",
    "cache",
    "ps",
    "add_model",
    "list_templates",
    "music",
]
