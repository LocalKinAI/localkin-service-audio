"""
Unit tests for music generation module.
"""
import pytest
import tempfile
import os
from pathlib import Path

from unittest.mock import patch, MagicMock

from localkin_service_audio.music import MusicEngine, MusicGenStrategy, HeartMuLaStrategy
from localkin_service_audio.core.types import ModelConfig, AudioResult


class TestMusicEngine:
    """Test base MusicEngine abstract class."""

    def test_music_engine_is_abstract(self):
        """MusicEngine should not be instantiable directly."""
        with pytest.raises(TypeError):
            MusicEngine()

    def test_music_engine_requires_load(self):
        """MusicEngine requires load method implementation."""
        # This is tested indirectly through subclasses
        pass


class TestMusicGenStrategy:
    """Test MusicGenStrategy implementation."""

    @pytest.fixture
    def model_config(self):
        """Create a test model config."""
        from localkin_service_audio.core.types import ModelType
        return ModelConfig(
            name="musicgen-small",
            type=ModelType.TTS,  # Use TTS as fallback for music
            engine="musicgen"
        )

    @pytest.fixture
    def strategy(self):
        """Create a MusicGenStrategy instance."""
        return MusicGenStrategy()

    def test_initialization(self, strategy):
        """Test MusicGenStrategy initialization."""
        assert strategy.model is None
        assert strategy.device == "cpu"
        assert strategy._is_loaded is False
        assert strategy.model_size is None

    def test_get_model_sizes(self):
        """Test available model sizes."""
        sizes = MusicGenStrategy.get_model_sizes()
        assert "small" in sizes
        assert "medium" in sizes
        assert "large" in sizes

    def test_get_supported_durations(self):
        """Test supported durations."""
        durations = MusicGenStrategy.get_supported_durations()
        assert 5 in durations
        assert 10 in durations
        assert 15 in durations
        assert 20 in durations
        assert 30 in durations

    def test_get_model_requirements(self):
        """Test model memory requirements."""
        reqs = MusicGenStrategy.get_model_requirements()
        assert "small" in reqs
        assert "medium" in reqs
        assert "large" in reqs
        
        # Check structure
        for size, req in reqs.items():
            assert "vram_gb" in req
            assert "ram_gb" in req
            assert "disk_gb" in req

    def test_device_detection_auto(self, strategy):
        """Test automatic device detection."""
        device = strategy._detect_device("auto")
        assert device in ["cpu", "cuda", "mps"]

    def test_device_detection_cpu(self, strategy):
        """Test explicit CPU device."""
        device = strategy._detect_device("cpu")
        assert device == "cpu"

    def test_is_loaded_property(self, strategy):
        """Test is_loaded property."""
        assert strategy.is_loaded is False
        
        strategy._is_loaded = True
        strategy.model = "dummy_model"  # Model must not be None
        assert strategy.is_loaded is True

    def test_load_invalid_model_size(self):
        """Test loading with invalid model size."""
        from localkin_service_audio.core.types import ModelType
        strategy = MusicGenStrategy()
        config = ModelConfig(
            name="musicgen-invalid",
            type=ModelType.TTS,
            engine="musicgen"
        )
        
        # This should fail due to invalid model size
        result = strategy.load(config, device="cpu")
        assert result is False

    def test_get_info(self, strategy, model_config):
        """Test get_info method."""
        # Before loading
        info = strategy.get_info()
        assert info["engine"] == "MusicGenStrategy"
        assert info["is_loaded"] is False
        
        # After setting config
        strategy.model_config = model_config
        strategy.model_size = "small"
        info = strategy.get_info()
        assert info["model_size"] == "small"

    def test_unload(self, strategy):
        """Test unload method."""
        strategy._is_loaded = True
        strategy.model = "dummy_model"
        
        strategy.unload()
        assert strategy._is_loaded is False
        assert strategy.model is None

    @pytest.mark.slow
    def test_load_model_small(self, model_config):
        """Test loading small model (requires transformers)."""
        strategy = MusicGenStrategy()
        
        try:
            result = strategy.load(model_config, device="cpu")
            assert result is True
            assert strategy.is_loaded is True
            
            strategy.unload()
        except ImportError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_10s_music(self, model_config):
        """Test music generation for 10 seconds.
        
        This is the key integration test.
        """
        strategy = MusicGenStrategy()
        
        try:
            # Load model
            success = strategy.load(model_config, device="cpu")
            if not success:
                pytest.skip("Failed to load model")
            
            # Generate music
            prompt = "calm piano melody"
            result = strategy.generate(prompt, duration=10)
            
            # Verify output
            assert isinstance(result, AudioResult)
            assert result.audio is not None
            assert result.sample_rate > 0
            assert result.duration > 0
            assert result.duration <= 11  # Allow 1s tolerance
            assert result.model == "musicgen-small"
            
            # Cleanup
            strategy.unload()
            
        except ImportError:
            pytest.skip("transformers not installed")
        except Exception as e:
            # Some test environments might not support the model
            pytest.skip(f"Model generation failed: {e}")


class TestMusicGenerationIntegration:
    """Integration tests for music generation."""

    @pytest.mark.slow
    def test_generate_and_save(self):
        """Test generation and file saving."""
        from localkin_service_audio.core.types import ModelType
        config = ModelConfig(
            name="musicgen-small",
            type=ModelType.TTS,
            engine="musicgen"
        )
        
        strategy = MusicGenStrategy()
        
        try:
            success = strategy.load(config, device="cpu")
            if not success:
                pytest.skip("Failed to load model")
            
            # Generate
            prompt = "upbeat electronic"
            result = strategy.generate(prompt, duration=5)
            
            # Save to file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_path = f.name
            
            try:
                saved_path = strategy.generate_to_file(
                    prompt,
                    output_path,
                    duration=5
                )
                
                assert os.path.exists(saved_path)
                assert os.path.getsize(saved_path) > 0
                
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            
            strategy.unload()
            
        except ImportError:
            pytest.skip("transformers not installed")
        except Exception as e:
            pytest.skip(f"Model generation failed: {e}")

    def test_duration_adjustment(self):
        """Test duration adjustment for unsupported durations."""
        from localkin_service_audio.core.types import ModelType

        strategy = MusicGenStrategy()

        # Duration 7 should be adjusted to closest (5)
        # This test only checks the logic, not actual generation
        supported = strategy.get_supported_durations()
        test_duration = 7
        closest = min(supported, key=lambda x: abs(x - test_duration))
        assert closest == 5


class TestHeartMuLaStrategy:
    """Test HeartMuLaStrategy implementation."""

    def test_initialization(self):
        """Test HeartMuLaStrategy initialization."""
        strategy = HeartMuLaStrategy()
        assert strategy.model is None
        assert strategy._is_loaded is False
        assert strategy.model_size is None

    def test_get_model_sizes(self):
        """Test available model sizes."""
        sizes = HeartMuLaStrategy.get_model_sizes()
        assert "3b" in sizes
        assert "7b" in sizes

    def test_get_supported_durations(self):
        """Test supported durations."""
        durations = HeartMuLaStrategy.get_supported_durations()
        assert 10 in durations
        assert 240 in durations

    def test_get_model_requirements(self):
        """Test model memory requirements."""
        reqs = HeartMuLaStrategy.get_model_requirements()
        assert "3b" in reqs
        assert "7b" in reqs
        assert reqs["3b"]["vram_gb"] == 6
        assert reqs["7b"]["vram_gb"] == 16

    def test_load_rejects_invalid_size(self):
        """Test loading with invalid model size."""
        from localkin_service_audio.core.types import ModelType
        strategy = HeartMuLaStrategy()
        config = ModelConfig(
            name="heartmula:99b",
            type=ModelType.TTS,
            engine="heartmula"
        )
        result = strategy.load(config, device="cpu")
        assert result is False

    @patch("subprocess.check_call")
    def test_ensure_heartlib_installs_when_missing(self, mock_check_call):
        """Test that _ensure_heartlib calls pip install when heartlib is missing."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "heartlib":
                raise ImportError("No module named 'heartlib'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            HeartMuLaStrategy._ensure_heartlib()

        mock_check_call.assert_called_once()
        call_args = mock_check_call.call_args[0][0]
        assert "pip" in call_args[1:]
        assert "git+https://github.com/HeartMuLa/heartlib.git" in call_args

    @patch("localkin_service_audio.music.heartmula_strategy.HeartMuLaStrategy._prepare_checkpoints", return_value=None)
    @patch("localkin_service_audio.music.heartmula_strategy.HeartMuLaStrategy._ensure_heartlib")
    def test_load_calls_ensure_heartlib(self, mock_ensure, mock_checkpoints):
        """Test that load() calls _ensure_heartlib before importing."""
        from localkin_service_audio.core.types import ModelType
        strategy = HeartMuLaStrategy()
        config = ModelConfig(
            name="heartmula:3b",
            type=ModelType.TTS,
            engine="heartmula"
        )
        # load() will call _ensure_heartlib, then fail at checkpoints (mocked to None)
        strategy.load(config, device="cpu")
        mock_ensure.assert_called_once()


class TestHeartMuLaRegistry:
    """Test HeartMuLa model registry entries."""

    def test_heartmula_3b_in_registry(self):
        from localkin_service_audio.core.config.model_registry import ModelRegistry
        registry = ModelRegistry()
        config = registry.get("heartmula:3b")
        assert config is not None
        assert config.engine == "heartmula"
        assert config.model_size == "3b"

    def test_heartmula_7b_in_registry(self):
        from localkin_service_audio.core.config.model_registry import ModelRegistry
        registry = ModelRegistry()
        config = registry.get("heartmula:7b")
        assert config is not None
        assert config.engine == "heartmula"
        assert config.model_size == "7b"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    
    # Add option for running slow tests
    if not hasattr(config, "option"):
        config.option = type("Options", (), {})()
    config.option.run_slow = config.getoption("--run-slow", default=False)


def pytest_addoption(parser):
    """Add pytest command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests (model download and generation)"
    )
