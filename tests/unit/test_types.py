"""
Tests for core types and dataclasses.
"""
import pytest
import numpy as np
import tempfile
import os


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic TranscriptionResult."""
        from localkin_service_audio.core.types import TranscriptionResult

        result = TranscriptionResult(text="Hello world")

        assert result.text == "Hello world"
        assert result.language is None
        assert result.segments is None
        assert result.emotion is None

    def test_full_creation(self):
        """Test creating TranscriptionResult with all fields."""
        from localkin_service_audio.core.types import TranscriptionResult, Segment

        segments = [
            Segment(text="Hello", start=0.0, end=0.5),
            Segment(text="world", start=0.5, end=1.0),
        ]

        result = TranscriptionResult(
            text="Hello world",
            language="en",
            language_probability=0.95,
            duration=1.0,
            segments=segments,
            emotion="happy",
            model="whisper:base",
            engine="whisper",
            processing_time=0.5,
        )

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert len(result.segments) == 2
        assert result.emotion == "happy"

    def test_str_representation(self):
        """Test string representation returns text."""
        from localkin_service_audio.core.types import TranscriptionResult

        result = TranscriptionResult(text="Test text")
        assert str(result) == "Test text"


class TestAudioResult:
    """Tests for AudioResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic AudioResult."""
        from localkin_service_audio.core.types import AudioResult

        audio = np.zeros(16000, dtype=np.float32)
        result = AudioResult(audio=audio, sample_rate=16000)

        assert len(result.audio) == 16000
        assert result.sample_rate == 16000

    def test_save_to_file(self):
        """Test saving audio to file."""
        from localkin_service_audio.core.types import AudioResult

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = AudioResult(audio=audio, sample_rate=16000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            saved_path = result.save(output_path)
            assert os.path.exists(saved_path)
            assert os.path.getsize(saved_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_to_wav_bytes(self):
        """Test converting to WAV bytes."""
        from localkin_service_audio.core.types import AudioResult

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = AudioResult(audio=audio, sample_rate=16000)

        wav_bytes = result.to_wav_bytes()

        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 0
        assert wav_bytes[:4] == b'RIFF'  # WAV header


class TestVoiceInfo:
    """Tests for VoiceInfo dataclass."""

    def test_basic_creation(self):
        """Test creating VoiceInfo."""
        from localkin_service_audio.core.types import VoiceInfo

        voice = VoiceInfo(
            id="af_bella",
            name="Bella",
            language="en",
            gender="female",
        )

        assert voice.id == "af_bella"
        assert voice.name == "Bella"
        assert voice.language == "en"
        assert voice.gender == "female"
        assert voice.is_cloned is False


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_basic_creation(self):
        """Test creating ModelConfig."""
        from localkin_service_audio.core.types import ModelConfig, ModelType

        config = ModelConfig(
            name="whisper:base",
            type=ModelType.STT,
            engine="whisper",
            model_size="base",
        )

        assert config.name == "whisper:base"
        assert config.type == ModelType.STT
        assert config.engine == "whisper"

    def test_supports_chinese(self):
        """Test Chinese language detection."""
        from localkin_service_audio.core.types import ModelConfig, ModelType

        # With zh in languages
        config1 = ModelConfig(
            name="test",
            type=ModelType.STT,
            engine="test",
            languages=["en", "zh"],
        )
        assert config1.supports_chinese is True

        # With chinese tag
        config2 = ModelConfig(
            name="test",
            type=ModelType.STT,
            engine="test",
            languages=["en"],
            tags=["chinese"],
        )
        assert config2.supports_chinese is True

        # Without Chinese
        config3 = ModelConfig(
            name="test",
            type=ModelType.STT,
            engine="test",
            languages=["en"],
        )
        assert config3.supports_chinese is False

    def test_supports_voice_cloning(self):
        """Test voice cloning feature detection."""
        from localkin_service_audio.core.types import ModelConfig, ModelType

        config = ModelConfig(
            name="test",
            type=ModelType.TTS,
            engine="test",
            features=["voice_cloning", "streaming"],
        )

        assert config.supports_voice_cloning is True
        assert config.supports_streaming is True


class TestSegment:
    """Tests for Segment dataclass."""

    def test_creation(self):
        """Test creating Segment."""
        from localkin_service_audio.core.types import Segment

        segment = Segment(
            text="Hello",
            start=0.0,
            end=0.5,
            confidence=0.95,
        )

        assert segment.text == "Hello"
        assert segment.start == 0.0
        assert segment.end == 0.5
        assert segment.confidence == 0.95


class TestHardwareProfile:
    """Tests for HardwareProfile dataclass."""

    def test_has_gpu(self):
        """Test GPU detection property."""
        from localkin_service_audio.core.types import HardwareProfile, DeviceType

        # CPU only
        cpu_profile = HardwareProfile(device=DeviceType.CPU)
        assert cpu_profile.has_gpu is False

        # CUDA
        cuda_profile = HardwareProfile(device=DeviceType.CUDA, vram_gb=8.0)
        assert cuda_profile.has_gpu is True

        # MPS
        mps_profile = HardwareProfile(device=DeviceType.MPS)
        assert mps_profile.has_gpu is True
