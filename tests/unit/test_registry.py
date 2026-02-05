"""
Tests for ModelRegistry.
"""
import pytest
import json
import tempfile
from pathlib import Path

from localkin_service_audio.core.types import ModelConfig, ModelType, HardwareRequirements
from localkin_service_audio.core.config.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry singleton."""

    def test_singleton_pattern(self):
        """Test ModelRegistry is a singleton."""
        # Reset singleton for testing
        ModelRegistry._instance = None

        registry1 = ModelRegistry()
        registry2 = ModelRegistry()

        assert registry1 is registry2

    def test_default_models_loaded(self):
        """Test default models are loaded."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        # Check some known models exist
        assert registry.get("whisper:base") is not None
        assert registry.get("whisper-cpp:base") is not None
        assert registry.get("kokoro") is not None

    def test_get_existing_model(self):
        """Test getting an existing model."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        model = registry.get("whisper:base")

        assert model is not None
        assert model.name == "whisper:base"
        assert model.type == ModelType.STT
        assert model.engine == "whisper"

    def test_get_nonexistent_model(self):
        """Test getting a nonexistent model."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        model = registry.get("nonexistent-model")

        assert model is None

    def test_register_custom_model(self):
        """Test registering a custom model."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        custom_config = ModelConfig(
            name="custom:test",
            type=ModelType.STT,
            engine="custom",
            model_size="test",
            languages=["en"],
            description="Custom test model",
        )

        registry.register("custom:test", custom_config)

        model = registry.get("custom:test")
        assert model is not None
        assert model.name == "custom:test"
        assert model.engine == "custom"

    def test_list_all(self):
        """Test listing all models."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        models = registry.list_all()

        assert len(models) > 0
        assert all(isinstance(m, ModelConfig) for m in models)

    def test_list_by_type_stt(self):
        """Test listing STT models."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        stt_models = registry.list_by_type(ModelType.STT)

        assert len(stt_models) > 0
        assert all(m.type == ModelType.STT for m in stt_models)

    def test_list_by_type_tts(self):
        """Test listing TTS models."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        tts_models = registry.list_by_type(ModelType.TTS)

        assert len(tts_models) > 0
        assert all(m.type == ModelType.TTS for m in tts_models)

    def test_list_stt_models_helper(self):
        """Test list_stt_models helper."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        stt_models = registry.list_stt_models()

        assert len(stt_models) > 0
        assert all(m.type == ModelType.STT for m in stt_models)

    def test_list_tts_models_helper(self):
        """Test list_tts_models helper."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        tts_models = registry.list_tts_models()

        assert len(tts_models) > 0
        assert all(m.type == ModelType.TTS for m in tts_models)

    def test_list_by_engine(self):
        """Test listing models by engine."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        whisper_models = registry.list_by_engine("whisper")

        assert len(whisper_models) > 0
        assert all(m.engine == "whisper" for m in whisper_models)

    def test_list_by_language(self):
        """Test listing models by language."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        english_models = registry.list_by_language("en")

        assert len(english_models) > 0
        assert all("en" in m.languages for m in english_models)

    def test_list_by_tag(self):
        """Test listing models by tag."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        fast_models = registry.list_by_tag("fast")

        assert len(fast_models) > 0
        assert all("fast" in m.tags for m in fast_models)

    def test_search_by_name(self):
        """Test searching models by name."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        results = registry.search("whisper")

        assert len(results) > 0
        # Search matches name, description, or tags - verify at least some have whisper in name
        whisper_named = [m for m in results if "whisper" in m.name.lower()]
        assert len(whisper_named) > 0

    def test_search_by_description(self):
        """Test searching models by description."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        results = registry.search("chinese")

        assert len(results) > 0

    def test_search_by_tag(self):
        """Test searching models by tag."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        results = registry.search("multilingual")

        assert len(results) > 0

    def test_search_no_results(self):
        """Test searching with no results."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        results = registry.search("xyz123nonexistent")

        assert len(results) == 0

    def test_to_dict(self):
        """Test exporting registry to dictionary."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        data = registry.to_dict()

        assert "stt_models" in data
        assert "tts_models" in data
        assert len(data["stt_models"]) > 0
        assert len(data["tts_models"]) > 0


class TestModelRegistryExternalConfig:
    """Tests for external config loading."""

    def test_load_json_config(self):
        """Test loading models from JSON config."""
        ModelRegistry._instance = None

        # Create a temporary JSON config
        config_data = {
            "models": {
                "test-model:v1": {
                    "type": "stt",
                    "engine": "test",
                    "model_size": "v1",
                    "languages": ["en", "zh"],
                    "description": "Test model from JSON",
                    "tags": ["test"],
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            registry = ModelRegistry()
            registry._load_config_file(Path(config_path))

            model = registry.get("test-model:v1")
            assert model is not None
            assert model.engine == "test"
            assert "zh" in model.languages
        finally:
            Path(config_path).unlink()

    def test_load_config_with_hardware_requirements(self):
        """Test loading config with hardware requirements."""
        ModelRegistry._instance = None

        config_data = {
            "models": {
                "gpu-model:large": {
                    "type": "stt",
                    "engine": "test",
                    "hardware_requirements": {
                        "min_vram_gb": 8.0,
                        "min_ram_gb": 16.0,
                    },
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            registry = ModelRegistry()
            registry._load_config_file(Path(config_path))

            model = registry.get("gpu-model:large")
            assert model is not None
            assert model.hardware_requirements is not None
            assert model.hardware_requirements.min_vram_gb == 8.0
        finally:
            Path(config_path).unlink()


class TestModelTypes:
    """Tests for model type categorization."""

    def test_whisper_variants_registered(self):
        """Test all Whisper variants are registered."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        variants = ["whisper:tiny", "whisper:base", "whisper:small", "whisper:medium"]
        for variant in variants:
            model = registry.get(variant)
            assert model is not None, f"Missing {variant}"
            assert model.engine == "whisper"

    def test_faster_whisper_variants_registered(self):
        """Test faster-whisper variants are registered."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        variants = ["faster-whisper:tiny", "faster-whisper:base"]
        for variant in variants:
            model = registry.get(variant)
            assert model is not None, f"Missing {variant}"
            assert model.engine == "faster-whisper"

    def test_whisper_cpp_variants_registered(self):
        """Test whisper-cpp variants are registered."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        variants = ["whisper-cpp:tiny", "whisper-cpp:base", "whisper-cpp:small"]
        for variant in variants:
            model = registry.get(variant)
            assert model is not None, f"Missing {variant}"
            assert model.engine == "whisper-cpp"

    def test_chinese_models_registered(self):
        """Test Chinese models are registered."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        assert registry.get("sensevoice:small") is not None
        assert registry.get("paraformer:zh") is not None
        assert registry.get("cosyvoice:300m") is not None
        assert registry.get("chattts") is not None

    def test_tts_models_have_voices(self):
        """Test TTS models with voices have them listed."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        kokoro = registry.get("kokoro")
        assert kokoro is not None
        assert kokoro.voices is not None
        assert len(kokoro.voices) > 0

    def test_models_with_features(self):
        """Test models have features listed."""
        ModelRegistry._instance = None
        registry = ModelRegistry()

        sensevoice = registry.get("sensevoice:small")
        assert sensevoice is not None
        assert "emotion" in sensevoice.features

        cosyvoice = registry.get("cosyvoice:300m")
        assert cosyvoice is not None
        assert "voice_cloning" in cosyvoice.features
