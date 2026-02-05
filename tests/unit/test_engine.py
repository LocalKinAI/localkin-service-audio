"""
Tests for AudioEngine and strategy pattern.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from localkin_service_audio.core.types import (
    TranscriptionResult,
    AudioResult,
    ModelConfig,
    ModelType,
    VoiceInfo,
)
from localkin_service_audio.core.audio_processing.engine import (
    AudioEngine,
    get_audio_engine,
)
from localkin_service_audio.core.audio_processing.stt.base import STTStrategy
from localkin_service_audio.core.audio_processing.tts.base import TTSStrategy


class MockSTTStrategy(STTStrategy):
    """Mock STT strategy for testing."""

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        self.model = "mock_model"
        self.model_config = model_config
        self.device = device
        self._is_loaded = True
        return True

    def transcribe(self, audio, language=None, **kwargs) -> TranscriptionResult:
        return TranscriptionResult(
            text="Mock transcription",
            language=language or "en",
            engine="mock",
        )


class MockTTSStrategy(TTSStrategy):
    """Mock TTS strategy for testing."""

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        self.model = "mock_model"
        self.model_config = model_config
        self.device = device
        self._is_loaded = True
        return True

    def synthesize(self, text, voice=None, speed=1.0, **kwargs) -> AudioResult:
        audio = np.zeros(16000, dtype=np.float32)
        return AudioResult(audio=audio, sample_rate=16000)

    def list_voices(self):
        return [
            VoiceInfo(id="voice1", name="Voice 1", language="en"),
            VoiceInfo(id="voice2", name="Voice 2", language="en"),
        ]


class TestAudioEngine:
    """Tests for AudioEngine facade."""

    def test_engine_initialization(self):
        """Test engine initializes with strategies registered."""
        engine = AudioEngine()

        # Should have registered strategies
        assert len(engine._stt_strategies) > 0
        assert len(engine._tts_strategies) > 0

        # Verify some expected engines are registered
        assert "whisper" in engine._stt_strategies
        assert "whisper-cpp" in engine._stt_strategies
        assert "kokoro" in engine._tts_strategies
        assert "native" in engine._tts_strategies

    def test_register_stt_strategy(self):
        """Test registering a custom STT strategy."""
        engine = AudioEngine()

        engine.register_stt_strategy("mock-stt", MockSTTStrategy)

        assert "mock-stt" in engine._stt_strategies
        assert engine._stt_strategies["mock-stt"] == MockSTTStrategy

    def test_register_tts_strategy(self):
        """Test registering a custom TTS strategy."""
        engine = AudioEngine()

        engine.register_tts_strategy("mock-tts", MockTTSStrategy)

        assert "mock-tts" in engine._tts_strategies
        assert engine._tts_strategies["mock-tts"] == MockTTSStrategy

    def test_parse_model_name_with_size(self):
        """Test parsing model name with size."""
        engine = AudioEngine()

        engine_name, size = engine._parse_model_name("whisper-cpp:base")

        assert engine_name == "whisper-cpp"
        assert size == "base"

    def test_parse_model_name_without_size(self):
        """Test parsing model name without size."""
        engine = AudioEngine()

        engine_name, size = engine._parse_model_name("kokoro")

        assert engine_name == "kokoro"
        assert size is None

    def test_load_stt_with_mock(self):
        """Test loading STT model with mock strategy."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock", MockSTTStrategy)

        success = engine.load_stt("mock:base", device="cpu")

        assert success is True
        assert engine._stt_model_name == "mock:base"
        assert engine._stt_strategy is not None
        assert engine._stt_strategy.is_loaded is True

    def test_load_stt_unknown_engine(self):
        """Test loading unknown STT engine raises error."""
        engine = AudioEngine()

        with pytest.raises(ValueError, match="Unknown STT engine"):
            engine.load_stt("unknown-engine:base")

    def test_transcribe_without_model(self):
        """Test transcribe without loading model raises error."""
        engine = AudioEngine()

        with pytest.raises(RuntimeError, match="No STT model loaded"):
            engine.transcribe(np.zeros(16000))

    def test_transcribe_with_mock(self):
        """Test transcription with mock strategy."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock", MockSTTStrategy)
        engine.load_stt("mock:base")

        result = engine.transcribe(np.zeros(16000), language="en")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Mock transcription"
        assert result.language == "en"

    def test_load_tts_with_mock(self):
        """Test loading TTS model with mock strategy."""
        engine = AudioEngine()
        engine.register_tts_strategy("mock", MockTTSStrategy)

        success = engine.load_tts("mock", device="cpu")

        assert success is True
        assert engine._tts_model_name == "mock"
        assert engine._tts_strategy is not None
        assert engine._tts_strategy.is_loaded is True

    def test_load_tts_unknown_engine(self):
        """Test loading unknown TTS engine raises error."""
        engine = AudioEngine()

        with pytest.raises(ValueError, match="Unknown TTS engine"):
            engine.load_tts("unknown-engine")

    def test_synthesize_without_model(self):
        """Test synthesize without loading model raises error."""
        engine = AudioEngine()

        with pytest.raises(RuntimeError, match="No TTS model loaded"):
            engine.synthesize("Hello world")

    def test_synthesize_with_mock(self):
        """Test synthesis with mock strategy."""
        engine = AudioEngine()
        engine.register_tts_strategy("mock", MockTTSStrategy)
        engine.load_tts("mock")

        result = engine.synthesize("Hello world")

        assert isinstance(result, AudioResult)
        assert len(result.audio) == 16000
        assert result.sample_rate == 16000

    def test_list_voices(self):
        """Test listing voices."""
        engine = AudioEngine()
        engine.register_tts_strategy("mock", MockTTSStrategy)
        engine.load_tts("mock")

        voices = engine.list_voices()

        assert len(voices) == 2
        assert voices[0].id == "voice1"
        assert voices[1].id == "voice2"

    def test_list_voices_no_model(self):
        """Test listing voices without model returns empty list."""
        engine = AudioEngine()

        voices = engine.list_voices()

        assert voices == []

    def test_unload_stt(self):
        """Test unloading STT model."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock", MockSTTStrategy)
        engine.load_stt("mock")

        engine.unload_stt()

        assert engine._stt_strategy is None
        assert engine._stt_model_name is None

    def test_unload_tts(self):
        """Test unloading TTS model."""
        engine = AudioEngine()
        engine.register_tts_strategy("mock", MockTTSStrategy)
        engine.load_tts("mock")

        engine.unload_tts()

        assert engine._tts_strategy is None
        assert engine._tts_model_name is None

    def test_unload_all(self):
        """Test unloading all models."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock", MockSTTStrategy)
        engine.register_tts_strategy("mock", MockTTSStrategy)
        engine.load_stt("mock")
        engine.load_tts("mock")

        engine.unload_all()

        assert engine._stt_strategy is None
        assert engine._tts_strategy is None

    def test_get_info(self):
        """Test getting engine info."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock", MockSTTStrategy)
        engine.register_tts_strategy("mock", MockTTSStrategy)
        engine.load_stt("mock")
        engine.load_tts("mock")

        info = engine.get_info()

        assert "stt" in info
        assert "tts" in info
        assert info["stt"]["model"] == "mock"
        assert info["stt"]["loaded"] is True
        assert info["tts"]["model"] == "mock"
        assert info["tts"]["loaded"] is True

    def test_list_engines(self):
        """Test listing available engines."""
        engine = AudioEngine()

        stt_engines = engine.list_stt_engines()
        tts_engines = engine.list_tts_engines()

        assert isinstance(stt_engines, list)
        assert isinstance(tts_engines, list)
        assert "whisper" in stt_engines
        assert "kokoro" in tts_engines

    def test_model_switching(self):
        """Test switching between models."""
        engine = AudioEngine()
        engine.register_stt_strategy("mock1", MockSTTStrategy)
        engine.register_stt_strategy("mock2", MockSTTStrategy)

        engine.load_stt("mock1")
        assert engine._stt_model_name == "mock1"

        engine.load_stt("mock2")
        assert engine._stt_model_name == "mock2"


class TestGlobalEngine:
    """Tests for global engine singleton."""

    def test_get_audio_engine_singleton(self):
        """Test get_audio_engine returns singleton."""
        engine1 = get_audio_engine()
        engine2 = get_audio_engine()

        assert engine1 is engine2


class TestSTTStrategyBase:
    """Tests for STTStrategy base class."""

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        strategy = MockSTTStrategy()

        assert strategy.is_loaded is False

        strategy.model = "test"
        strategy._is_loaded = True

        assert strategy.is_loaded is True

    def test_unload(self):
        """Test unload method."""
        strategy = MockSTTStrategy()
        strategy.model = "test"
        strategy._is_loaded = True

        strategy.unload()

        assert strategy.model is None
        assert strategy._is_loaded is False

    def test_get_info(self):
        """Test get_info method."""
        strategy = MockSTTStrategy()
        config = ModelConfig(
            name="test",
            type=ModelType.STT,
            engine="mock",
        )
        strategy.model_config = config
        strategy.device = "cpu"

        info = strategy.get_info()

        assert info["engine"] == "MockSTTStrategy"
        assert info["device"] == "cpu"
        assert info["is_loaded"] is False

    def test_detect_device_auto(self):
        """Test device detection with auto."""
        strategy = MockSTTStrategy()

        # Without torch/CUDA, should fall back to CPU
        device = strategy._detect_device("auto")
        assert device in ["cpu", "cuda", "mps"]

    def test_detect_device_explicit(self):
        """Test device detection with explicit value."""
        strategy = MockSTTStrategy()

        device = strategy._detect_device("cpu")
        assert device == "cpu"


class TestTTSStrategyBase:
    """Tests for TTSStrategy base class."""

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        strategy = MockTTSStrategy()

        assert strategy.is_loaded is False

        strategy.model = "test"
        strategy._is_loaded = True

        assert strategy.is_loaded is True

    def test_unload(self):
        """Test unload method."""
        strategy = MockTTSStrategy()
        strategy.model = "test"
        strategy._is_loaded = True
        strategy._current_voice = "voice1"

        strategy.unload()

        assert strategy.model is None
        assert strategy._is_loaded is False
        assert strategy._current_voice is None

    def test_set_voice_valid(self):
        """Test setting valid voice."""
        strategy = MockTTSStrategy()
        strategy.model = "test"
        strategy._is_loaded = True

        result = strategy.set_voice("voice1")

        assert result is True
        assert strategy._current_voice == "voice1"

    def test_set_voice_invalid(self):
        """Test setting invalid voice."""
        strategy = MockTTSStrategy()
        strategy.model = "test"
        strategy._is_loaded = True

        result = strategy.set_voice("invalid-voice")

        assert result is False

    def test_clone_voice_not_implemented(self):
        """Test clone_voice raises NotImplementedError by default."""
        strategy = MockTTSStrategy()

        with pytest.raises(NotImplementedError):
            strategy.clone_voice("audio.wav", "Hello")

    def test_get_info(self):
        """Test get_info method."""
        strategy = MockTTSStrategy()
        strategy.model = "test"
        strategy._is_loaded = True
        config = ModelConfig(
            name="test",
            type=ModelType.TTS,
            engine="mock",
        )
        strategy.model_config = config
        strategy.device = "cpu"

        info = strategy.get_info()

        assert info["engine"] == "MockTTSStrategy"
        assert info["device"] == "cpu"
        assert info["is_loaded"] is True
        assert "voice1" in info["voices"]
