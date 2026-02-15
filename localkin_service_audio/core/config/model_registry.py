"""
Model Registry - Centralized model configuration management.

Like ollamadiffuser's model_registry, supports three-tier loading:
1. Hardcoded defaults (built-in models)
2. External config files (~/.localkin-service-audio/models.yaml)
3. Remote API (future)
"""
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os

from ..types import ModelConfig, ModelType, HardwareRequirements


class ModelRegistry:
    """
    Centralized registry for audio models.

    Manages model configurations from multiple sources:
    - Built-in defaults
    - User config files (JSON/YAML)
    - Environment variables
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._models: Dict[str, ModelConfig] = {}
        self._load_default_models()
        self._load_external_models()
        self._initialized = True

    def _load_default_models(self):
        """Load built-in model definitions."""
        # STT Models
        stt_models = {
            # Whisper variants
            "whisper:tiny": ModelConfig(
                name="whisper:tiny",
                type=ModelType.STT,
                engine="whisper",
                model_size="tiny",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="OpenAI Whisper tiny - fastest, lowest accuracy",
                tags=["multilingual", "fast"],
            ),
            "whisper:base": ModelConfig(
                name="whisper:base",
                type=ModelType.STT,
                engine="whisper",
                model_size="base",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="OpenAI Whisper base - balanced speed/accuracy",
                tags=["multilingual"],
            ),
            "whisper:small": ModelConfig(
                name="whisper:small",
                type=ModelType.STT,
                engine="whisper",
                model_size="small",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="OpenAI Whisper small - good accuracy",
                tags=["multilingual"],
            ),
            "whisper:medium": ModelConfig(
                name="whisper:medium",
                type=ModelType.STT,
                engine="whisper",
                model_size="medium",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="OpenAI Whisper medium - high accuracy",
                tags=["multilingual", "accurate"],
            ),
            "whisper:large-v3": ModelConfig(
                name="whisper:large-v3",
                type=ModelType.STT,
                engine="whisper",
                model_size="large-v3",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="OpenAI Whisper large-v3 - best accuracy",
                tags=["multilingual", "accurate", "large"],
                hardware_requirements=HardwareRequirements(min_vram_gb=6),
            ),

            # Faster-whisper variants
            "faster-whisper:tiny": ModelConfig(
                name="faster-whisper:tiny",
                type=ModelType.STT,
                engine="faster-whisper",
                model_size="tiny",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Faster-whisper tiny - 4x faster than OpenAI",
                tags=["multilingual", "fast"],
            ),
            "faster-whisper:base": ModelConfig(
                name="faster-whisper:base",
                type=ModelType.STT,
                engine="faster-whisper",
                model_size="base",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Faster-whisper base - 4x faster, balanced",
                tags=["multilingual"],
            ),
            "faster-whisper:large-v3": ModelConfig(
                name="faster-whisper:large-v3",
                type=ModelType.STT,
                engine="faster-whisper",
                model_size="large-v3",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Faster-whisper large-v3 - fast and accurate",
                tags=["multilingual", "accurate", "fast"],
                hardware_requirements=HardwareRequirements(min_vram_gb=4),
            ),
            "faster-whisper:turbo": ModelConfig(
                name="faster-whisper:turbo",
                type=ModelType.STT,
                engine="faster-whisper",
                model_size="turbo",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Faster-whisper turbo - optimized speed",
                tags=["multilingual", "fast"],
            ),
            "faster-whisper:distil-large-v3": ModelConfig(
                name="faster-whisper:distil-large-v3",
                type=ModelType.STT,
                engine="faster-whisper",
                model_size="distil-large-v3",
                languages=["en"],
                description="Distil-Whisper - 6x faster, English focused",
                tags=["english", "fast", "distilled"],
            ),

            # Whisper.cpp variants
            "whisper-cpp:tiny": ModelConfig(
                name="whisper-cpp:tiny",
                type=ModelType.STT,
                engine="whisper-cpp",
                model_size="tiny",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Whisper.cpp tiny - ultra-fast CPU inference",
                tags=["multilingual", "ultra-fast", "cpu"],
            ),
            "whisper-cpp:base": ModelConfig(
                name="whisper-cpp:base",
                type=ModelType.STT,
                engine="whisper-cpp",
                model_size="base",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Whisper.cpp base - fast CPU inference",
                tags=["multilingual", "fast", "cpu"],
            ),
            "whisper-cpp:small": ModelConfig(
                name="whisper-cpp:small",
                type=ModelType.STT,
                engine="whisper-cpp",
                model_size="small",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Whisper.cpp small - balanced for CPU",
                tags=["multilingual", "cpu"],
            ),
            "whisper-cpp:medium": ModelConfig(
                name="whisper-cpp:medium",
                type=ModelType.STT,
                engine="whisper-cpp",
                model_size="medium",
                languages=["en", "zh", "de", "es", "fr", "ja", "ko", "pt", "ru"],
                description="Whisper.cpp medium - good accuracy on CPU",
                tags=["multilingual", "accurate", "cpu"],
            ),

            # Moonshine - ultra-fast English STT
            "moonshine:tiny": ModelConfig(
                name="moonshine:tiny",
                type=ModelType.STT,
                engine="moonshine",
                model_size="tiny",
                repo_id="usefulsensors/moonshine-tiny",
                languages=["en"],
                description="Moonshine tiny - ultra-fast real-time ASR (~20MB)",
                tags=["english", "ultra-fast", "lightweight", "cpu"],
            ),
            "moonshine:base": ModelConfig(
                name="moonshine:base",
                type=ModelType.STT,
                engine="moonshine",
                model_size="base",
                repo_id="usefulsensors/moonshine-base",
                languages=["en"],
                description="Moonshine base - 5x real-time on CPU",
                tags=["english", "fast", "cpu"],
            ),

            # Chinese / Multilingual STT
            "sensevoice:small": ModelConfig(
                name="sensevoice:small",
                type=ModelType.STT,
                engine="sensevoice",
                model_size="small",
                repo_id="FunAudioLLM/SenseVoiceSmall",
                languages=["zh", "en", "ja", "ko", "yue"],
                features=["emotion", "language_detection", "audio_events"],
                description="SenseVoice Small - 15x faster than Whisper, emotion detection",
                tags=["chinese", "multilingual", "fast", "emotion"],
            ),
            "paraformer:zh": ModelConfig(
                name="paraformer:zh",
                type=ModelType.STT,
                engine="funasr",
                model_size="paraformer-zh",
                repo_id="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                languages=["zh"],
                description="Paraformer Chinese - Alibaba fast Chinese ASR",
                tags=["chinese", "fast", "accurate"],
            ),

            # NVIDIA Models (planned)
            "parakeet:1.1b": ModelConfig(
                name="parakeet:1.1b",
                type=ModelType.STT,
                engine="parakeet",
                model_size="1.1b",
                repo_id="nvidia/parakeet-tdt-1.1b",
                languages=["en"],
                description="Parakeet TDT 1.1B - NVIDIA fastest ASR (>2000x RT)",
                tags=["english", "nvidia", "ultra-fast"],
            ),
            "canary:1b": ModelConfig(
                name="canary:1b",
                type=ModelType.STT,
                engine="canary",
                model_size="1b",
                repo_id="nvidia/canary-1b",
                languages=["en", "de", "es", "fr"],
                description="Canary 1B - NVIDIA #1 on HuggingFace leaderboard",
                tags=["multilingual", "nvidia", "accurate"],
            ),
        }

        # TTS Models
        tts_models = {
            "native": ModelConfig(
                name="native",
                type=ModelType.TTS,
                engine="native",
                description="Native OS TTS - no download required",
                tags=["fast", "offline"],
            ),
            "kokoro": ModelConfig(
                name="kokoro",
                type=ModelType.TTS,
                engine="kokoro",
                model_size="82m",
                languages=["en", "es", "fr", "hi", "it", "ja", "pt", "zh"],
                voices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bm_george", "zf_xiaoxiao", "jf_alpha"],
                description="Kokoro TTS - high-quality multilingual neural TTS",
                tags=["multilingual", "neural", "quality"],
            ),
            # Chinese TTS
            "cosyvoice:300m": ModelConfig(
                name="cosyvoice:300m",
                type=ModelType.TTS,
                engine="cosyvoice",
                model_size="300m-sft",
                repo_id="iic/CosyVoice-300M-SFT",
                languages=["zh", "en", "ja", "ko", "yue"],
                features=["voice_cloning", "streaming", "cross_lingual"],
                voices=["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男"],
                description="CosyVoice 300M - Alibaba SOTA Chinese TTS",
                tags=["chinese", "voice-cloning", "streaming", "sota"],
            ),
            "chattts": ModelConfig(
                name="chattts",
                type=ModelType.TTS,
                engine="chattts",
                repo_id="2noise/ChatTTS",
                languages=["zh", "en"],
                features=["conversational", "emotion", "laughter"],
                description="ChatTTS - conversational TTS for dialogue",
                tags=["chinese", "english", "conversational", "emotional"],
            ),
            "gpt-sovits": ModelConfig(
                name="gpt-sovits",
                type=ModelType.TTS,
                engine="gpt-sovits",
                repo_id="lj1995/GPT-SoVITS",
                languages=["zh", "en", "ja"],
                features=["voice_cloning", "fine_tuning"],
                description="GPT-SoVITS - voice cloning with 5s audio",
                tags=["chinese", "japanese", "voice-cloning"],
            ),
            "f5-tts": ModelConfig(
                name="f5-tts",
                type=ModelType.TTS,
                engine="f5-tts",
                repo_id="SWivid/F5-TTS",
                languages=["en", "zh"],
                features=["voice_cloning", "zero_shot"],
                description="F5-TTS - zero-shot voice cloning",
                tags=["voice-cloning", "zero-shot"],
            ),
            "parler-tts": ModelConfig(
                name="parler-tts",
                type=ModelType.TTS,
                engine="parler",
                repo_id="parler-tts/parler-tts-large-v1",
                languages=["en"],
                features=["text_described_voice"],
                description="Parler TTS - describe voice with natural language",
                tags=["english", "text-described"],
            ),
            # Music Generation Models (HeartMuLa)
            "heartmula:3b": ModelConfig(
                name="heartmula:3b",
                type=ModelType.TTS,
                engine="heartmula",
                model_size="3b",
                repo_id="HeartMuLa/HeartMuLa-oss-3B",
                languages=["en", "zh", "ja", "ko", "es"],
                description="HeartMuLa 3B - multilingual music generation with lyrics & tags",
                tags=["music", "multilingual", "chinese", "lyrics"],
                hardware_requirements=HardwareRequirements(min_vram_gb=6),
            ),
            "heartmula:7b": ModelConfig(
                name="heartmula:7b",
                type=ModelType.TTS,
                engine="heartmula",
                model_size="7b",
                repo_id="HeartMuLa/HeartMuLa-oss-7B",
                languages=["en", "zh", "ja", "ko", "es"],
                description="HeartMuLa 7B - higher quality multilingual music generation",
                tags=["music", "multilingual", "chinese", "lyrics"],
                hardware_requirements=HardwareRequirements(min_vram_gb=16),
            ),
        }

        # Merge all models
        self._models.update(stt_models)
        self._models.update(tts_models)

    def _load_external_models(self):
        """Load models from external config files."""
        from .settings import _default_home
        config_paths = [
            _default_home() / "models.json",
            _default_home() / "models.yaml",
        ]

        # Add environment variable path if set
        env_config = os.environ.get("LOCALKIN_MODEL_CONFIG")
        if env_config:
            config_paths.append(Path(env_config))

        for path in config_paths:
            if path and path.exists():
                try:
                    self._load_config_file(path)
                except Exception as e:
                    print(f"Warning: Failed to load models from {path}: {e}")

    def _load_config_file(self, path: Path):
        """Load models from a config file."""
        if path.suffix == ".yaml" or path.suffix == ".yml":
            try:
                import yaml
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                print("Warning: PyYAML not installed. Skipping YAML config.")
                return
        else:
            with open(path, "r") as f:
                data = json.load(f)

        # Process models
        for model_type in ["stt_models", "tts_models", "models"]:
            if model_type in data:
                for name, config in data[model_type].items():
                    self._register_from_dict(name, config)

    def _register_from_dict(self, name: str, config: Dict[str, Any]):
        """Register a model from a dictionary config."""
        # Determine model type
        model_type = config.get("type", "stt")
        if isinstance(model_type, str):
            model_type = ModelType.STT if model_type.lower() == "stt" else ModelType.TTS

        # Parse hardware requirements
        hw_req = None
        if "hardware_requirements" in config:
            hw_req = HardwareRequirements(**config["hardware_requirements"])

        # Create ModelConfig
        model_config = ModelConfig(
            name=name,
            type=model_type,
            engine=config.get("engine", ""),
            repo_id=config.get("repo_id"),
            model_size=config.get("model_size"),
            languages=config.get("languages", ["en"]),
            features=config.get("features", []),
            hardware_requirements=hw_req,
            parameters=config.get("parameters", {}),
            pipeline_class=config.get("pipeline_class"),
            license=config.get("license"),
            description=config.get("description"),
            tags=config.get("tags", []),
            voices=config.get("voices"),
        )

        self._models[name] = model_config

    def register(self, name: str, config: ModelConfig):
        """Register a model at runtime."""
        self._models[name] = config

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a model config by name."""
        # Try exact match first
        if name in self._models:
            return self._models[name]

        # Try without engine prefix (e.g., "base" -> "whisper-cpp:base")
        for key, config in self._models.items():
            if config.model_size == name:
                return config

        return None

    def list_all(self) -> List[ModelConfig]:
        """List all registered models."""
        return list(self._models.values())

    def list_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        """List models by type (STT or TTS)."""
        return [m for m in self._models.values() if m.type == model_type]

    def list_stt_models(self) -> List[ModelConfig]:
        """List STT models."""
        return self.list_by_type(ModelType.STT)

    def list_tts_models(self) -> List[ModelConfig]:
        """List TTS models."""
        return self.list_by_type(ModelType.TTS)

    def list_by_engine(self, engine: str) -> List[ModelConfig]:
        """List models by engine."""
        return [m for m in self._models.values() if m.engine == engine]

    def list_by_language(self, language: str) -> List[ModelConfig]:
        """List models supporting a language."""
        return [m for m in self._models.values() if language in m.languages]

    def list_by_tag(self, tag: str) -> List[ModelConfig]:
        """List models with a specific tag."""
        return [m for m in self._models.values() if tag in m.tags]

    def search(self, query: str) -> List[ModelConfig]:
        """Search models by name, description, or tags."""
        query = query.lower()
        results = []

        for model in self._models.values():
            if (
                query in model.name.lower() or
                (model.description and query in model.description.lower()) or
                any(query in tag.lower() for tag in model.tags)
            ):
                results.append(model)

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dictionary."""
        return {
            "stt_models": {
                name: asdict(config)
                for name, config in self._models.items()
                if config.type == ModelType.STT
            },
            "tts_models": {
                name: asdict(config)
                for name, config in self._models.items()
                if config.type == ModelType.TTS
            },
        }


# Global registry instance (singleton)
model_registry = ModelRegistry()
