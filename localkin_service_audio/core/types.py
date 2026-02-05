"""
Core types and dataclasses for LocalKin Audio.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import numpy as np


class ModelType(str, Enum):
    """Type of audio model."""
    STT = "stt"
    TTS = "tts"


class DeviceType(str, Enum):
    """Compute device type."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class Segment:
    """A transcription segment with timing information."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    segments: Optional[List[Segment]] = None

    # Extended features (SenseVoice, etc.)
    emotion: Optional[str] = None
    audio_events: Optional[List[str]] = None

    # Metadata
    model: Optional[str] = None
    engine: Optional[str] = None
    processing_time: Optional[float] = None

    def __str__(self) -> str:
        return self.text


@dataclass
class AudioResult:
    """Result from text-to-speech synthesis."""
    audio: np.ndarray
    sample_rate: int = 22050

    # Metadata
    model: Optional[str] = None
    voice: Optional[str] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None

    def save(self, path: str, format: str = "wav") -> str:
        """Save audio to file."""
        import soundfile as sf
        sf.write(path, self.audio, self.sample_rate)
        return path

    def to_wav_bytes(self) -> bytes:
        """Convert to WAV bytes."""
        import io
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, self.audio, self.sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()


@dataclass
class VoiceInfo:
    """Information about a TTS voice."""
    id: str
    name: str
    language: str = "en"
    gender: Optional[str] = None
    description: Optional[str] = None
    sample_rate: int = 22050

    # For voice cloning
    is_cloned: bool = False
    reference_audio: Optional[str] = None


@dataclass
class HardwareRequirements:
    """Hardware requirements for a model."""
    min_vram_gb: Optional[float] = None
    recommended_vram_gb: Optional[float] = None
    min_ram_gb: Optional[float] = None
    supported_devices: List[str] = field(default_factory=lambda: ["cpu", "cuda", "mps"])
    performance_notes: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for an audio model."""
    name: str
    type: ModelType
    engine: str  # whisper, faster-whisper, whisper-cpp, sensevoice, kokoro, cosyvoice, etc.

    # Model source
    repo_id: Optional[str] = None
    model_size: Optional[str] = None
    local_path: Optional[str] = None

    # Languages supported
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Features
    features: List[str] = field(default_factory=list)  # voice_cloning, streaming, emotion, etc.

    # Hardware requirements
    hardware_requirements: Optional[HardwareRequirements] = None

    # Engine-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For generic strategy - specify pipeline class
    pipeline_class: Optional[str] = None

    # Metadata
    license: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Voices (for TTS)
    voices: Optional[List[str]] = None

    @property
    def supports_chinese(self) -> bool:
        return "zh" in self.languages or "chinese" in self.tags

    @property
    def supports_voice_cloning(self) -> bool:
        return "voice_cloning" in self.features

    @property
    def supports_streaming(self) -> bool:
        return "streaming" in self.features


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    device: DeviceType
    vram_gb: float = 0.0
    ram_gb: float = 0.0
    cpu_cores: int = 1
    gpu_name: Optional[str] = None

    @property
    def has_gpu(self) -> bool:
        return self.device in (DeviceType.CUDA, DeviceType.MPS)
