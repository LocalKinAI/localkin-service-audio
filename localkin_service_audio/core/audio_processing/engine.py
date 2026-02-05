"""
AudioEngine - Unified facade for audio processing operations.

This is the main entry point for STT/TTS operations, following the
Facade pattern from ollamadiffuser's InferenceEngine.
"""
from typing import Optional, Dict, Any, Union, List, Type
import numpy as np

from ..types import (
    TranscriptionResult,
    AudioResult,
    VoiceInfo,
    ModelConfig,
    ModelType,
)
from .stt.base import STTStrategy
from .tts.base import TTSStrategy


class AudioEngine:
    """
    Unified facade for audio processing operations.

    Delegates STT/TTS operations to appropriate strategies based on
    model configuration. Maintains loaded models and provides a
    consistent interface regardless of the underlying engine.

    Example usage:
        engine = AudioEngine()
        engine.load_stt("whisper-cpp:base")
        result = engine.transcribe("audio.wav")
        print(result.text)

        engine.load_tts("kokoro")
        audio = engine.synthesize("Hello world")
        audio.save("output.wav")
    """

    # Strategy registries
    _stt_strategies: Dict[str, Type[STTStrategy]] = {}
    _tts_strategies: Dict[str, Type[TTSStrategy]] = {}

    def __init__(self):
        self._stt_strategy: Optional[STTStrategy] = None
        self._tts_strategy: Optional[TTSStrategy] = None
        self._stt_model_name: Optional[str] = None
        self._tts_model_name: Optional[str] = None

        # Register default strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register built-in strategies."""
        # ==================== STT Strategies ====================
        # Whisper family
        from .stt.whisper_strategy import WhisperStrategy
        from .stt.faster_whisper_strategy import FasterWhisperStrategy
        from .stt.whisper_cpp_strategy import WhisperCppStrategy

        self.register_stt_strategy("whisper", WhisperStrategy)
        self.register_stt_strategy("openai-whisper", WhisperStrategy)
        self.register_stt_strategy("faster-whisper", FasterWhisperStrategy)
        self.register_stt_strategy("whisper-cpp", WhisperCppStrategy)

        # Chinese / Multilingual (FunASR)
        from .stt.sensevoice_strategy import SenseVoiceStrategy
        from .stt.paraformer_strategy import ParaformerStrategy

        self.register_stt_strategy("sensevoice", SenseVoiceStrategy)
        self.register_stt_strategy("funasr", ParaformerStrategy)
        self.register_stt_strategy("paraformer", ParaformerStrategy)

        # Fast English
        from .stt.moonshine_strategy import MoonshineStrategy

        self.register_stt_strategy("moonshine", MoonshineStrategy)

        # ==================== TTS Strategies ====================
        # Basic
        from .tts.native_strategy import NativeStrategy
        from .tts.kokoro_strategy import KokoroStrategy

        self.register_tts_strategy("native", NativeStrategy)
        self.register_tts_strategy("pyttsx3", NativeStrategy)
        self.register_tts_strategy("kokoro", KokoroStrategy)

        # Chinese TTS
        from .tts.cosyvoice_strategy import CosyVoiceStrategy
        from .tts.chattts_strategy import ChatTTSStrategy

        self.register_tts_strategy("cosyvoice", CosyVoiceStrategy)
        self.register_tts_strategy("chattts", ChatTTSStrategy)

        # Voice Cloning
        from .tts.f5_strategy import F5TTSStrategy

        self.register_tts_strategy("f5-tts", F5TTSStrategy)
        self.register_tts_strategy("f5", F5TTSStrategy)

    @classmethod
    def register_stt_strategy(cls, engine_name: str, strategy_class: Type[STTStrategy]):
        """Register an STT strategy for an engine name."""
        cls._stt_strategies[engine_name.lower()] = strategy_class

    @classmethod
    def register_tts_strategy(cls, engine_name: str, strategy_class: Type[TTSStrategy]):
        """Register a TTS strategy for an engine name."""
        cls._tts_strategies[engine_name.lower()] = strategy_class

    # ==================== STT Operations ====================

    def load_stt(
        self,
        model_name: str,
        device: str = "auto",
        **kwargs
    ) -> bool:
        """
        Load an STT model.

        Args:
            model_name: Model name in format "engine:size" or just "engine"
                        e.g., "whisper-cpp:base", "faster-whisper:large-v3"
            device: Device to use ("auto", "cpu", "cuda", "mps")
            **kwargs: Additional model configuration

        Returns:
            True if loaded successfully
        """
        # Parse model name
        engine, model_size = self._parse_model_name(model_name)

        # Get strategy class
        strategy_class = self._get_stt_strategy_class(engine)
        if not strategy_class:
            raise ValueError(f"Unknown STT engine: {engine}")

        # Create model config
        config = ModelConfig(
            name=model_name,
            type=ModelType.STT,
            engine=engine,
            model_size=model_size,
            **kwargs
        )

        # Unload existing model if different
        if self._stt_strategy and self._stt_model_name != model_name:
            self.unload_stt()

        # Create and load strategy
        self._stt_strategy = strategy_class()
        success = self._stt_strategy.load(config, device)

        if success:
            self._stt_model_name = model_name

        return success

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (numpy array at 16kHz) or path to audio file
            language: Language code (e.g., "en", "zh") or None for auto-detect
            **kwargs: Engine-specific arguments

        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        if not self._stt_strategy or not self._stt_strategy.is_loaded:
            raise RuntimeError("No STT model loaded. Call load_stt() first.")

        return self._stt_strategy.transcribe(audio, language=language, **kwargs)

    def transcribe_file(self, audio_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe an audio file."""
        return self.transcribe(audio_path, **kwargs)

    def transcribe_stream(self, audio_stream, **kwargs):
        """Transcribe streaming audio (generator)."""
        if not self._stt_strategy:
            raise RuntimeError("No STT model loaded.")
        return self._stt_strategy.transcribe_stream(audio_stream, **kwargs)

    def unload_stt(self):
        """Unload the current STT model."""
        if self._stt_strategy:
            self._stt_strategy.unload()
            self._stt_strategy = None
            self._stt_model_name = None

    # ==================== TTS Operations ====================

    def load_tts(
        self,
        model_name: str,
        device: str = "auto",
        **kwargs
    ) -> bool:
        """
        Load a TTS model.

        Args:
            model_name: Model name in format "engine:variant" or just "engine"
                        e.g., "kokoro", "cosyvoice:300m-sft"
            device: Device to use
            **kwargs: Additional model configuration

        Returns:
            True if loaded successfully
        """
        # Parse model name
        engine, model_size = self._parse_model_name(model_name)

        # Get strategy class
        strategy_class = self._get_tts_strategy_class(engine)
        if not strategy_class:
            raise ValueError(f"Unknown TTS engine: {engine}")

        # Create model config
        config = ModelConfig(
            name=model_name,
            type=ModelType.TTS,
            engine=engine,
            model_size=model_size,
            **kwargs
        )

        # Unload existing model if different
        if self._tts_strategy and self._tts_model_name != model_name:
            self.unload_tts()

        # Create and load strategy
        self._tts_strategy = strategy_class()
        success = self._tts_strategy.load(config, device)

        if success:
            self._tts_model_name = model_name

        return success

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID to use (None for default)
            speed: Speech speed multiplier
            **kwargs: Engine-specific arguments

        Returns:
            AudioResult with synthesized audio
        """
        if not self._tts_strategy or not self._tts_strategy.is_loaded:
            raise RuntimeError("No TTS model loaded. Call load_tts() first.")

        return self._tts_strategy.synthesize(text, voice=voice, speed=speed, **kwargs)

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        **kwargs
    ) -> str:
        """Synthesize speech and save to file."""
        result = self.synthesize(text, voice=voice, **kwargs)
        return result.save(output_path)

    def synthesize_stream(self, text: str, **kwargs):
        """Synthesize speech as a stream (generator)."""
        if not self._tts_strategy:
            raise RuntimeError("No TTS model loaded.")
        return self._tts_strategy.synthesize_stream(text, **kwargs)

    def clone_voice(
        self,
        reference_audio: Union[str, np.ndarray],
        text: str,
        reference_text: Optional[str] = None,
        **kwargs
    ) -> AudioResult:
        """
        Clone a voice from reference audio and synthesize text.

        Args:
            reference_audio: Path to reference audio or audio array
            text: Text to synthesize with cloned voice
            reference_text: Transcript of reference audio (for some models)

        Returns:
            AudioResult with synthesized audio
        """
        if not self._tts_strategy:
            raise RuntimeError("No TTS model loaded.")
        return self._tts_strategy.clone_voice(
            reference_audio, text, reference_text, **kwargs
        )

    def list_voices(self) -> List[VoiceInfo]:
        """List available voices for the current TTS model."""
        if not self._tts_strategy:
            return []
        return self._tts_strategy.list_voices()

    def set_voice(self, voice: str) -> bool:
        """Set the current voice."""
        if not self._tts_strategy:
            return False
        return self._tts_strategy.set_voice(voice)

    def unload_tts(self):
        """Unload the current TTS model."""
        if self._tts_strategy:
            self._tts_strategy.unload()
            self._tts_strategy = None
            self._tts_model_name = None

    # ==================== Utility Methods ====================

    def unload_all(self):
        """Unload all models."""
        self.unload_stt()
        self.unload_tts()

    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "stt": {
                "model": self._stt_model_name,
                "loaded": self._stt_strategy.is_loaded if self._stt_strategy else False,
                "info": self._stt_strategy.get_info() if self._stt_strategy else None,
            },
            "tts": {
                "model": self._tts_model_name,
                "loaded": self._tts_strategy.is_loaded if self._tts_strategy else False,
                "info": self._tts_strategy.get_info() if self._tts_strategy else None,
            },
        }

    @classmethod
    def list_stt_engines(cls) -> List[str]:
        """List available STT engine names."""
        return list(cls._stt_strategies.keys())

    @classmethod
    def list_tts_engines(cls) -> List[str]:
        """List available TTS engine names."""
        return list(cls._tts_strategies.keys())

    # ==================== Private Methods ====================

    def _parse_model_name(self, model_name: str) -> tuple:
        """
        Parse model name into engine and size.

        Args:
            model_name: e.g., "whisper-cpp:base", "kokoro", "faster-whisper:large-v3"

        Returns:
            (engine, size) tuple
        """
        if ":" in model_name:
            parts = model_name.split(":", 1)
            return parts[0].lower(), parts[1]
        return model_name.lower(), None

    def _get_stt_strategy_class(self, engine: str) -> Optional[Type[STTStrategy]]:
        """Get STT strategy class for engine name."""
        return self._stt_strategies.get(engine.lower())

    def _get_tts_strategy_class(self, engine: str) -> Optional[Type[TTSStrategy]]:
        """Get TTS strategy class for engine name."""
        return self._tts_strategies.get(engine.lower())


# Global engine instance (singleton pattern like ollamadiffuser)
_engine_instance: Optional[AudioEngine] = None


def get_audio_engine() -> AudioEngine:
    """Get the global AudioEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = AudioEngine()
    return _engine_instance


# Convenience functions
def transcribe(audio: Union[np.ndarray, str], model: str = "whisper-cpp:base", **kwargs) -> TranscriptionResult:
    """Quick transcription using default engine."""
    engine = get_audio_engine()
    if not engine._stt_strategy or engine._stt_model_name != model:
        engine.load_stt(model)
    return engine.transcribe(audio, **kwargs)


def synthesize(text: str, model: str = "kokoro", voice: Optional[str] = None, **kwargs) -> AudioResult:
    """Quick synthesis using default engine."""
    engine = get_audio_engine()
    if not engine._tts_strategy or engine._tts_model_name != model:
        engine.load_tts(model)
    return engine.synthesize(text, voice=voice, **kwargs)
