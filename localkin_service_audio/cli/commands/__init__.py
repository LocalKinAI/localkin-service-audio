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
]
