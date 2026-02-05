"""
Pytest configuration and fixtures for LocalKin Audio tests.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_audio_array():
    """Generate a simple test audio array (1 second of sine wave)."""
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio, sample_rate


@pytest.fixture
def sample_audio_file(sample_audio_array):
    """Create a temporary WAV file with test audio."""
    import soundfile as sf

    audio, sample_rate = sample_audio_array

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        yield f.name

    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def sample_speech_audio_file():
    """Get the included sample audio file if available."""
    from localkin_service_audio import get_sample_audio_path

    sample_path = get_sample_audio_path()
    if sample_path and os.path.exists(sample_path):
        return sample_path

    pytest.skip("Sample audio file not available")


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_config():
    """Create a mock model configuration."""
    from localkin_service_audio.core.types import ModelConfig, ModelType

    return ModelConfig(
        name="test-model:base",
        type=ModelType.STT,
        engine="test",
        model_size="base",
        languages=["en", "zh"],
        description="Test model for unit tests",
        tags=["test", "mock"],
    )


@pytest.fixture
def audio_engine():
    """Get a fresh AudioEngine instance."""
    from localkin_service_audio.core.audio_processing import AudioEngine
    return AudioEngine()


@pytest.fixture
def model_registry():
    """Get the model registry instance."""
    from localkin_service_audio.core.config import model_registry
    return model_registry
