"""
Application settings and configuration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json


def _default_home() -> Path:
    """Get the base directory, respecting LOCALKIN_HOME."""
    return Path(os.environ.get("LOCALKIN_HOME", Path.home() / ".localkin-service-audio"))


@dataclass
class Settings:
    """Application settings."""

    # Paths
    cache_dir: Path = field(default_factory=lambda: _default_home() / "cache")
    config_dir: Path = field(default_factory=lambda: _default_home())
    models_dir: Path = field(default_factory=lambda: _default_home() / "models")

    # Default models
    default_stt_model: str = "whisper-cpp:base"
    default_tts_model: str = "kokoro"

    # Server settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Device settings
    default_device: str = "auto"

    def __post_init__(self):
        """Ensure directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config file."""
        if config_path is None:
            config_path = _default_home() / "config.json"

        settings = cls()

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)

                for key, value in data.items():
                    if hasattr(settings, key):
                        if key.endswith("_dir"):
                            setattr(settings, key, Path(value))
                        else:
                            setattr(settings, key, value)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        # Override with environment variables
        env_mapping = {
            "LOCALKIN_CACHE_DIR": "cache_dir",
            "LOCALKIN_CONFIG_DIR": "config_dir",
            "LOCALKIN_MODELS_DIR": "models_dir",
            "LOCALKIN_DEFAULT_STT": "default_stt_model",
            "LOCALKIN_DEFAULT_TTS": "default_tts_model",
            "LOCALKIN_API_HOST": "api_host",
            "LOCALKIN_API_PORT": "api_port",
            "LOCALKIN_DEVICE": "default_device",
        }

        for env_var, attr in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                if attr.endswith("_dir"):
                    setattr(settings, attr, Path(value))
                elif attr == "api_port":
                    setattr(settings, attr, int(value))
                else:
                    setattr(settings, attr, value)

        return settings

    def save(self, config_path: Optional[Path] = None):
        """Save settings to config file."""
        if config_path is None:
            config_path = self.config_dir / "config.json"

        data = {
            "cache_dir": str(self.cache_dir),
            "config_dir": str(self.config_dir),
            "models_dir": str(self.models_dir),
            "default_stt_model": self.default_stt_model,
            "default_tts_model": self.default_tts_model,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "default_device": self.default_device,
        }

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_model_path(self, model_name: str) -> Path:
        """Get path for a model."""
        return self.models_dir / model_name.replace(":", "_").replace("/", "_")


# Global settings instance
settings = Settings.load()
