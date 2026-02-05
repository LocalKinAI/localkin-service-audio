# LocalKin Service Audio - Upgrade Plan

## Overview

Apply architectural patterns from ollamadiffuser to modernize and extend localkin-service-audio.

**Current State:** v1.1.3, monolithic CLI (2,133 lines), basic model registry
**Target State:** v2.0.0, modular architecture with Strategy Pattern, extensible registry, MCP support

---

## Phase 1: Core Architecture Refactoring

### 1.1 Strategy Pattern for Audio Engines

Replace monolithic audio processing with Strategy Pattern like ollamadiffuser's inference engine.

**New Structure:**
```
core/
├── audio_processing/
│   ├── engine.py              # Facade - delegates to strategies
│   ├── base.py                # Abstract base classes
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py            # STTStrategy ABC
│   │   ├── whisper_strategy.py
│   │   ├── faster_whisper_strategy.py
│   │   ├── whisper_cpp_strategy.py
│   │   ├── moonshine_strategy.py      # NEW
│   │   ├── distil_whisper_strategy.py # NEW
│   │   └── generic_strategy.py        # Extensibility
│   └── tts/
│       ├── __init__.py
│       ├── base.py            # TTSStrategy ABC
│       ├── native_strategy.py
│       ├── kokoro_strategy.py
│       ├── xtts_strategy.py
│       ├── f5_strategy.py             # NEW
│       ├── parler_strategy.py         # NEW
│       └── generic_strategy.py        # Extensibility
```

**Base Strategy Interface:**
```python
# core/audio_processing/stt/base.py
class STTStrategy(ABC):
    @abstractmethod
    def load(self, model_config: ModelConfig, device: str) -> bool:
        """Load the model into memory."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        """Transcribe audio to text."""
        pass

    def unload(self) -> None:
        """Release model resources."""
        pass

    def get_info(self) -> Dict:
        """Return model information."""
        pass

# core/audio_processing/tts/base.py
class TTSStrategy(ABC):
    @abstractmethod
    def load(self, model_config: ModelConfig, device: str) -> bool:
        pass

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> AudioResult:
        pass

    def unload(self) -> None:
        pass

    def list_voices(self) -> List[VoiceInfo]:
        pass
```

**AudioEngine Facade:**
```python
# core/audio_processing/engine.py
class AudioEngine:
    """Unified facade for STT/TTS operations."""

    def __init__(self):
        self._stt_strategy: Optional[STTStrategy] = None
        self._tts_strategy: Optional[TTSStrategy] = None

    def load_stt(self, model_name: str, device: str = "auto") -> bool:
        model_config = model_registry.get(model_name)
        self._stt_strategy = self._get_stt_strategy(model_config.engine_type)
        return self._stt_strategy.load(model_config, device)

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        return self._stt_strategy.transcribe(audio, **kwargs)

    # Similar for TTS...
```

### 1.2 Enhanced Model Registry

Like ollamadiffuser, support three-tier model loading:

1. **Hardcoded defaults** - Built-in models
2. **External config files** - User-defined models (`~/.localkin-service-audio/models.yaml`)
3. **Remote API** - Fetch from registry API (future)

**New Registry Structure:**
```python
# core/config/model_registry.py
class ModelRegistry:
    _instance = None

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._load_default_models()
        self._load_external_models()

    def _load_default_models(self):
        """Load from core/models.json"""
        pass

    def _load_external_models(self):
        """Load from user config directory"""
        config_paths = [
            Path.home() / ".localkin-service-audio" / "models.yaml",
            Path.home() / ".localkin-service-audio" / "models.json",
            os.environ.get("LOCALKIN_MODEL_CONFIG"),
        ]
        for path in config_paths:
            if path and Path(path).exists():
                self._merge_models(self._load_config(path))

    def register(self, name: str, config: ModelConfig):
        """Runtime model registration."""
        self._models[name] = config

    def get(self, name: str) -> ModelConfig:
        return self._models.get(name)

    def list_by_type(self, model_type: str) -> List[ModelConfig]:
        return [m for m in self._models.values() if m.type == model_type]

# Singleton instance
model_registry = ModelRegistry()
```

**Enhanced Model Config:**
```python
@dataclass
class ModelConfig:
    name: str
    type: Literal["stt", "tts"]
    engine: str  # whisper, faster-whisper, whisper-cpp, kokoro, etc.
    repo_id: Optional[str] = None
    model_size: Optional[str] = None

    # Hardware requirements (from ollamadiffuser)
    hardware_requirements: Optional[HardwareRequirements] = None

    # Engine-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # For generic strategy
    pipeline_class: Optional[str] = None

    # Metadata
    license: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
```

---

## Phase 2: CLI Restructuring

### 2.1 Command Separation by Domain

Break the 2,133-line cli.py into focused modules:

```
cli/
├── __init__.py
├── main.py                 # Entry point, Click group, mode switching
├── commands/
│   ├── __init__.py
│   ├── transcribe.py       # kin audio transcribe
│   ├── synthesize.py       # kin audio tts
│   ├── listen.py           # kin audio listen (real-time)
│   ├── models.py           # kin audio models, pull, rm
│   ├── serve.py            # kin audio run (API server)
│   └── config.py           # kin audio config
├── utils/
│   ├── __init__.py
│   ├── progress.py         # OllamaStyleProgress display
│   ├── device.py           # Hardware detection
│   └── audio.py            # Audio I/O helpers
└── recommend.py            # Hardware-aware model recommendations
```

**Main Entry Point:**
```python
# cli/main.py
import click
from .commands import transcribe, synthesize, listen, models, serve, config

@click.group()
@click.option("--mode", type=click.Choice(["cli", "api", "ui", "mcp"]), default="cli")
@click.pass_context
def cli(ctx, mode):
    """LocalKin Audio - Voice AI Platform"""
    ctx.ensure_object(dict)
    ctx.obj["mode"] = mode

@cli.group()
def audio():
    """Audio processing commands."""
    pass

# Register command groups
audio.add_command(transcribe.transcribe)
audio.add_command(synthesize.tts)
audio.add_command(listen.listen)
audio.add_command(models.models)
audio.add_command(models.pull)
audio.add_command(serve.run)
audio.add_command(config.config)

@cli.command()
def web():
    """Start web interface."""
    from ..ui import create_app
    create_app().run()

@cli.command()
def mcp():
    """Start MCP server for Claude integration."""
    from ..mcp import run_server
    run_server()
```

### 2.2 Progress Display (Ollama Style)

```python
# cli/utils/progress.py
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

class OllamaStyleProgress:
    """Mimics Ollama's clean progress display."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

    def download_model(self, model_name: str, callback=None):
        with self.progress:
            task = self.progress.add_task(f"Pulling {model_name}...", total=100)
            # Update via callback

    def transcribe_progress(self, audio_duration: float):
        # Real-time transcription progress
        pass
```

---

## Phase 3: New Models Integration

### 3.1 STT Models - General Purpose

| Model | Engine Type | Priority | Speed | Notes |
|-------|-------------|----------|-------|-------|
| Moonshine | moonshine | High | 5x RT | Ultra-fast, tiny footprint, English |
| Distil-Whisper | distil-whisper | High | 6x faster | Distilled from Whisper large-v3 |
| Whisper Large V3 Turbo | faster-whisper | High | 8x RT | Latest OpenAI, best quality |
| Parakeet TDT 1.1B | parakeet | High | >2000x RT | NVIDIA, fastest on Open ASR |
| Canary Qwen 2.5B | canary | Medium | Fast | NVIDIA, #1 on HF leaderboard (5.63% WER) |
| Canary 1B Flash | canary | Medium | Very Fast | Speed-optimized variant |

### 3.2 STT Models - Chinese / Multilingual 🇨🇳

| Model | Engine Type | Priority | Languages | Notes |
|-------|-------------|----------|-----------|-------|
| **SenseVoice** | sensevoice | **High** | 50+ (CN/EN/JP/KO) | 15x faster than Whisper, emotion detection |
| **Paraformer-zh** | funasr | **High** | Chinese | Alibaba, non-autoregressive, very fast |
| **FunASR** | funasr | High | CN/EN | Full toolkit: ASR + VAD + punctuation |
| **FireRedASR** | firered | Medium | CN + dialects | SOTA on Mandarin benchmarks |
| Fun-ASR-Nano | funasr | Medium | 31 languages | Lightweight, real-time |
| Whisper (Chinese) | whisper | Medium | Multilingual | Good Chinese but slower |

**SenseVoice Strategy Example (Chinese):**
```python
# core/audio_processing/stt/sensevoice_strategy.py
class SenseVoiceStrategy(STTStrategy):
    """Alibaba SenseVoice - 15x faster than Whisper, multilingual."""

    MODELS = {
        "sensevoice-small": "FunAudioLLM/SenseVoiceSmall",
        "sensevoice-large": "FunAudioLLM/SenseVoiceLarge",
    }

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from funasr import AutoModel
        self.model = AutoModel(
            model=self.MODELS.get(model_config.model_size, "sensevoice-small"),
            device=device,
        )
        return True

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        result = self.model.generate(audio)
        return TranscriptionResult(
            text=result["text"],
            language=result.get("language", "zh"),
            emotion=result.get("emotion"),  # SenseVoice detects emotion
        )
```

**Paraformer Strategy Example (Chinese):**
```python
# core/audio_processing/stt/paraformer_strategy.py
class ParaformerStrategy(STTStrategy):
    """Alibaba Paraformer - Fast non-autoregressive Chinese ASR."""

    MODELS = {
        "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "paraformer-en": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    }

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from funasr import AutoModel
        model_id = self.MODELS.get(model_config.model_size, "paraformer-zh")
        self.model = AutoModel(model=model_id, device=device)
        # Optional: load VAD and punctuation models
        self.vad = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
        self.punc = AutoModel(model="iic/punc_ct-transformer_cn-en-common-vocab471067-large")
        return True

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        # VAD -> ASR -> Punctuation pipeline
        segments = self.vad.generate(audio)
        text = self.model.generate(audio, segments=segments)
        text_with_punc = self.punc.generate(text)
        return TranscriptionResult(text=text_with_punc, language="zh")
```

**Moonshine Strategy Example:**
```python
# core/audio_processing/stt/moonshine_strategy.py
class MoonshineStrategy(STTStrategy):
    """Ultra-fast STT using Moonshine models."""

    MODELS = {
        "moonshine-tiny": "usefulsensors/moonshine-tiny",
        "moonshine-base": "usefulsensors/moonshine-base",
    }

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from moonshine import Moonshine
        self.model = Moonshine(model_config.model_size)
        return True

    def transcribe(self, audio: np.ndarray, **kwargs) -> TranscriptionResult:
        text = self.model.transcribe(audio)
        return TranscriptionResult(text=text, language="en")
```

---

### 3.3 TTS Models - General Purpose

| Model | Engine Type | Priority | Features | Notes |
|-------|-------------|----------|----------|-------|
| F5-TTS | f5-tts | **High** | Voice cloning | Zero-shot, high quality |
| Kokoro | kokoro | High | Fast, quality | Already supported |
| Parler-TTS | parler | High | Text-described | "A young woman with soft voice" |
| OuteTTS | outetss | Medium | Lightweight | Fast inference |
| MeloTTS | melo | Medium | Multilingual | EN/ES/FR/ZH/JP/KR |
| Fish Speech | fish | Medium | High quality | Good prosody |
| XTTS v2 | xtts | Medium | Voice cloning | Already supported |

### 3.4 TTS Models - Chinese 🇨🇳

| Model | Engine Type | Priority | Features | Notes |
|-------|-------------|----------|----------|-------|
| **CosyVoice 2** | cosyvoice | **High** | Voice cloning, streaming | Alibaba, SOTA Chinese TTS |
| **ChatTTS** | chattts | **High** | Conversational | Natural dialogue, EN/CN |
| **GPT-SoVITS v3** | gpt-sovits | **High** | Voice cloning | CN/JP/EN, 5s reference audio |
| **EmotiVoice** | emotivoice | Medium | Emotional | NetEase, emotion control |
| Bert-VITS2 | bert-vits2 | Medium | High quality | Chinese focus |
| PaddleSpeech | paddlespeech | Medium | Full toolkit | Baidu, many voices |
| Muyan-TTS | muyan | Low | Podcast | Optimized for long-form |

**CosyVoice 2 Strategy Example (Chinese):**
```python
# core/audio_processing/tts/cosyvoice_strategy.py
class CosyVoiceStrategy(TTSStrategy):
    """Alibaba CosyVoice 2 - Best Chinese TTS with voice cloning."""

    MODELS = {
        "cosyvoice-300m": "iic/CosyVoice-300M",
        "cosyvoice-300m-sft": "iic/CosyVoice-300M-SFT",
        "cosyvoice-300m-instruct": "iic/CosyVoice-300M-Instruct",
        "cosyvoice-ttsfrd": "iic/CosyVoice-ttsfrd",
    }

    BUILTIN_VOICES = [
        "中文女", "中文男", "日语男", "粤语女", "英文女", "英文男", "韩语女"
    ]

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from cosyvoice import CosyVoice
        model_id = self.MODELS.get(model_config.model_size, "cosyvoice-300m-sft")
        self.model = CosyVoice(model_id)
        return True

    def synthesize(self, text: str, voice: str = "中文女", **kwargs) -> AudioResult:
        # Use built-in voice
        audio = self.model.inference_sft(text, voice)
        return AudioResult(audio=audio, sample_rate=22050)

    def clone_voice(self, reference_audio: str, text: str) -> AudioResult:
        """Zero-shot voice cloning."""
        audio = self.model.inference_zero_shot(
            text,
            prompt_audio=reference_audio,
        )
        return AudioResult(audio=audio, sample_rate=22050)

    def cross_lingual(self, text: str, reference_audio: str) -> AudioResult:
        """Cross-lingual synthesis (speak Chinese with English voice)."""
        audio = self.model.inference_cross_lingual(text, prompt_audio=reference_audio)
        return AudioResult(audio=audio, sample_rate=22050)

    def list_voices(self) -> List[VoiceInfo]:
        return [VoiceInfo(name=v, language="zh") for v in self.BUILTIN_VOICES]
```

**ChatTTS Strategy Example:**
```python
# core/audio_processing/tts/chattts_strategy.py
class ChatTTSStrategy(TTSStrategy):
    """ChatTTS - Conversational TTS for dialogue applications."""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        import ChatTTS
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Set compile=True for faster inference
        return True

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        # Generate with random speaker
        params_infer = ChatTTS.Chat.InferCodeParams(
            spk_emb=None,  # Random speaker
            temperature=0.3,
            top_P=0.7,
            top_K=20,
        )
        wavs = self.model.infer(text, params_infer_code=params_infer)
        return AudioResult(audio=wavs[0], sample_rate=24000)

    def synthesize_with_emotion(self, text: str, emotion: str = "happy") -> AudioResult:
        """Synthesize with emotion control."""
        # ChatTTS supports [laugh], [break] tokens
        emotional_text = self._add_emotion_tokens(text, emotion)
        return self.synthesize(emotional_text)
```

**GPT-SoVITS Strategy Example:**
```python
# core/audio_processing/tts/gpt_sovits_strategy.py
class GPTSoVITSStrategy(TTSStrategy):
    """GPT-SoVITS v3 - Voice cloning with just 5 seconds of audio."""

    LANGUAGES = ["zh", "en", "ja"]

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from GPT_SoVITS import TTS
        self.model = TTS(device=device)
        return True

    def clone_voice(
        self,
        reference_audio: str,
        reference_text: str,  # Transcript of reference audio
        target_text: str,
        language: str = "zh"
    ) -> AudioResult:
        """Clone voice from reference audio (5-10 seconds recommended)."""
        audio = self.model.synthesize(
            ref_audio=reference_audio,
            ref_text=reference_text,
            target_text=target_text,
            language=language,
        )
        return AudioResult(audio=audio, sample_rate=32000)

    def train_voice(self, audio_samples: List[str], name: str) -> str:
        """Fine-tune model on voice samples for better quality."""
        # Returns path to fine-tuned model
        pass
```

**F5-TTS Strategy Example:**
```python
# core/audio_processing/tts/f5_strategy.py
class F5TTSStrategy(TTSStrategy):
    """Zero-shot voice cloning with F5-TTS."""

    def load(self, model_config: ModelConfig, device: str) -> bool:
        from f5_tts import F5TTS
        self.model = F5TTS.from_pretrained(model_config.repo_id)
        self.model.to(device)
        return True

    def synthesize(self, text: str, reference_audio: Optional[str] = None, **kwargs) -> AudioResult:
        if reference_audio:
            # Voice cloning mode
            audio = self.model.generate(text, reference=reference_audio)
        else:
            audio = self.model.generate(text)
        return AudioResult(audio=audio, sample_rate=24000)

    def clone_voice(self, reference_audio: str, text: str) -> AudioResult:
        """Clone voice from reference audio."""
        return self.synthesize(text, reference_audio=reference_audio)
```

---

### 3.5 Complete Model Registry (models.json additions)

```json
{
  "stt_models": {
    "sensevoice-small": {
      "name": "SenseVoice Small",
      "type": "stt",
      "engine": "sensevoice",
      "repo_id": "FunAudioLLM/SenseVoiceSmall",
      "languages": ["zh", "en", "ja", "ko", "yue"],
      "features": ["emotion", "language_detection", "audio_events"],
      "speed": "15x faster than Whisper",
      "description": "Alibaba multilingual ASR with emotion detection",
      "tags": ["chinese", "multilingual", "fast", "emotion"]
    },
    "paraformer-zh": {
      "name": "Paraformer Chinese",
      "type": "stt",
      "engine": "funasr",
      "repo_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
      "languages": ["zh"],
      "description": "Alibaba non-autoregressive Chinese ASR",
      "tags": ["chinese", "fast", "accurate"]
    },
    "firered-asr": {
      "name": "FireRedASR",
      "type": "stt",
      "engine": "firered",
      "repo_id": "FireRedTeam/FireRedASR",
      "languages": ["zh", "zh-dialects", "en"],
      "description": "SOTA Mandarin ASR with dialect support",
      "tags": ["chinese", "dialects", "sota"]
    },
    "moonshine-base": {
      "name": "Moonshine Base",
      "type": "stt",
      "engine": "moonshine",
      "repo_id": "usefulsensors/moonshine-base",
      "languages": ["en"],
      "description": "Ultra-fast English ASR",
      "tags": ["english", "ultra-fast", "lightweight"]
    },
    "parakeet-tdt-1.1b": {
      "name": "Parakeet TDT 1.1B",
      "type": "stt",
      "engine": "parakeet",
      "repo_id": "nvidia/parakeet-tdt-1.1b",
      "languages": ["en"],
      "speed": ">2000x realtime",
      "description": "NVIDIA fastest open ASR model",
      "tags": ["english", "nvidia", "ultra-fast"]
    },
    "canary-1b": {
      "name": "Canary 1B",
      "type": "stt",
      "engine": "canary",
      "repo_id": "nvidia/canary-1b",
      "languages": ["en", "de", "es", "fr"],
      "description": "NVIDIA multilingual, #1 on HuggingFace leaderboard",
      "tags": ["multilingual", "nvidia", "accurate"]
    }
  },
  "tts_models": {
    "cosyvoice-300m-sft": {
      "name": "CosyVoice 300M",
      "type": "tts",
      "engine": "cosyvoice",
      "repo_id": "iic/CosyVoice-300M-SFT",
      "languages": ["zh", "en", "ja", "ko", "yue"],
      "features": ["voice_cloning", "streaming", "cross_lingual"],
      "voices": ["中文女", "中文男", "日语男", "粤语女", "英文女", "英文男"],
      "description": "Alibaba SOTA Chinese TTS with voice cloning",
      "tags": ["chinese", "voice-cloning", "streaming", "sota"]
    },
    "chattts": {
      "name": "ChatTTS",
      "type": "tts",
      "engine": "chattts",
      "repo_id": "2noise/ChatTTS",
      "languages": ["zh", "en"],
      "features": ["conversational", "emotion", "laughter"],
      "description": "Conversational TTS for dialogue applications",
      "tags": ["chinese", "english", "conversational", "emotional"]
    },
    "gpt-sovits-v3": {
      "name": "GPT-SoVITS v3",
      "type": "tts",
      "engine": "gpt-sovits",
      "repo_id": "lj1995/GPT-SoVITS",
      "languages": ["zh", "en", "ja"],
      "features": ["voice_cloning", "fine_tuning"],
      "description": "Voice cloning with 5 seconds of audio",
      "tags": ["chinese", "japanese", "voice-cloning"]
    },
    "emotivoice": {
      "name": "EmotiVoice",
      "type": "tts",
      "engine": "emotivoice",
      "repo_id": "netease-youdao/emotivoice",
      "languages": ["zh", "en"],
      "features": ["emotion_control"],
      "emotions": ["happy", "sad", "angry", "surprised", "fearful"],
      "description": "NetEase emotional TTS",
      "tags": ["chinese", "emotional"]
    },
    "f5-tts": {
      "name": "F5-TTS",
      "type": "tts",
      "engine": "f5-tts",
      "repo_id": "SWivid/F5-TTS",
      "languages": ["en", "zh"],
      "features": ["voice_cloning", "zero_shot"],
      "description": "Zero-shot voice cloning",
      "tags": ["voice-cloning", "zero-shot"]
    },
    "parler-tts": {
      "name": "Parler TTS",
      "type": "tts",
      "engine": "parler",
      "repo_id": "parler-tts/parler-tts-large-v1",
      "languages": ["en"],
      "features": ["text_described_voice"],
      "description": "Describe voice with natural language",
      "tags": ["english", "text-described"]
    },
    "melo-tts": {
      "name": "MeloTTS",
      "type": "tts",
      "engine": "melo",
      "repo_id": "myshell-ai/MeloTTS",
      "languages": ["en", "es", "fr", "zh", "ja", "ko"],
      "description": "Multilingual high-quality TTS",
      "tags": ["multilingual"]
    }
  }
}
```

---

### 3.6 Strategy Directory Structure (Updated)

```
core/audio_processing/
├── engine.py
├── stt/
│   ├── __init__.py
│   ├── base.py
│   ├── whisper_strategy.py          # OpenAI Whisper
│   ├── faster_whisper_strategy.py   # CTranslate2 Whisper
│   ├── whisper_cpp_strategy.py      # whisper.cpp
│   ├── moonshine_strategy.py        # NEW - Ultra-fast
│   ├── parakeet_strategy.py         # NEW - NVIDIA
│   ├── canary_strategy.py           # NEW - NVIDIA
│   ├── sensevoice_strategy.py       # NEW - Chinese/Multilingual
│   ├── paraformer_strategy.py       # NEW - Chinese
│   ├── funasr_strategy.py           # NEW - FunASR toolkit
│   ├── firered_strategy.py          # NEW - Chinese dialects
│   └── generic_strategy.py
└── tts/
    ├── __init__.py
    ├── base.py
    ├── native_strategy.py            # pyttsx3
    ├── kokoro_strategy.py            # Kokoro
    ├── xtts_strategy.py              # Coqui XTTS v2
    ├── f5_strategy.py                # NEW - Voice cloning
    ├── parler_strategy.py            # NEW - Text-described
    ├── cosyvoice_strategy.py         # NEW - Chinese SOTA
    ├── chattts_strategy.py           # NEW - Conversational
    ├── gpt_sovits_strategy.py        # NEW - Voice cloning
    ├── emotivoice_strategy.py        # NEW - Emotional
    ├── melo_strategy.py              # NEW - Multilingual
    └── generic_strategy.py
```

---

## Phase 4: New Features

### 4.1 MCP Server (Claude Integration)

Like ollamadiffuser, add MCP server for Claude Code/Desktop integration:

```
mcp/
├── __init__.py
└── server.py
```

```python
# mcp/server.py
from mcp.server import Server
from mcp.types import Tool

mcp_server = Server("localkin-audio")

@mcp_server.tool()
async def transcribe_audio(audio_path: str, engine: str = "whisper-cpp") -> str:
    """Transcribe audio file to text."""
    from ..core import audio_engine
    result = audio_engine.transcribe_file(audio_path, engine=engine)
    return result.text

@mcp_server.tool()
async def synthesize_speech(text: str, voice: str = "default", model: str = "kokoro") -> str:
    """Convert text to speech, returns path to audio file."""
    from ..core import audio_engine
    result = audio_engine.synthesize(text, voice=voice, model=model)
    output_path = f"/tmp/tts_{uuid.uuid4()}.wav"
    result.save(output_path)
    return output_path

@mcp_server.tool()
async def list_voices(model: str = "kokoro") -> list:
    """List available voices for TTS model."""
    from ..core import audio_engine
    return audio_engine.list_voices(model)

def run_server():
    import asyncio
    asyncio.run(mcp_server.run())
```

### 4.2 Hardware-Aware Recommendations

```python
# cli/recommend.py
import torch
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    device: str  # cuda, mps, cpu
    vram_gb: float
    ram_gb: float
    cpu_cores: int

def detect_hardware() -> HardwareProfile:
    """Detect system hardware capabilities."""
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return HardwareProfile("cuda", vram, get_ram(), os.cpu_count())
    elif torch.backends.mps.is_available():
        # Apple Silicon - estimate unified memory
        return HardwareProfile("mps", get_ram() * 0.7, get_ram(), os.cpu_count())
    else:
        return HardwareProfile("cpu", 0, get_ram(), os.cpu_count())

def recommend_models(hardware: HardwareProfile) -> Dict[str, List[str]]:
    """Recommend STT/TTS models based on hardware."""
    recommendations = {"stt": [], "tts": []}

    if hardware.vram_gb >= 8:
        recommendations["stt"].extend(["whisper-large-v3", "faster-whisper-large-v3"])
        recommendations["tts"].extend(["xtts-v2", "f5-tts"])
    elif hardware.vram_gb >= 4:
        recommendations["stt"].extend(["faster-whisper-medium", "distil-whisper"])
        recommendations["tts"].extend(["kokoro-82m", "parler-tts"])
    else:
        recommendations["stt"].extend(["whisper-cpp-tiny", "moonshine-tiny"])
        recommendations["tts"].extend(["native", "kokoro-82m"])

    return recommendations
```

### 4.3 Voice Cloning Support

```python
# core/audio_processing/voice_cloning.py
class VoiceCloner:
    """Voice cloning using F5-TTS or XTTS."""

    def __init__(self, model: str = "f5-tts"):
        self.model = model
        self._strategy = self._get_strategy(model)

    def clone(self, reference_audio: str, text: str, output_path: str) -> str:
        """Clone voice from reference and synthesize text."""
        result = self._strategy.clone_voice(reference_audio, text)
        result.save(output_path)
        return output_path

    def create_voice_profile(self, audio_samples: List[str], name: str) -> VoiceProfile:
        """Create reusable voice profile from multiple samples."""
        # Combine samples, extract voice embedding
        pass
```

### 4.4 Streaming Transcription (WebSocket)

```python
# api/websocket.py
from fastapi import WebSocket
import asyncio

class StreamingTranscriber:
    """Real-time streaming transcription via WebSocket."""

    async def handle_stream(self, websocket: WebSocket):
        await websocket.accept()

        audio_buffer = AudioBuffer(sample_rate=16000)

        while True:
            data = await websocket.receive_bytes()
            audio_buffer.add(data)

            if audio_buffer.has_speech():
                # Transcribe chunk
                text = await self._transcribe_chunk(audio_buffer.get_chunk())
                await websocket.send_json({
                    "type": "partial",
                    "text": text
                })

            if audio_buffer.is_end_of_speech():
                # Final transcription
                final_text = await self._transcribe_final(audio_buffer.get_all())
                await websocket.send_json({
                    "type": "final",
                    "text": final_text
                })
                audio_buffer.clear()
```

---

## Phase 5: API Enhancements

### 5.1 Unified API Design

```python
# api/server.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="LocalKin Audio API", version="2.0.0")

# Request/Response Models
class TranscribeRequest(BaseModel):
    engine: str = "whisper-cpp"
    model_size: str = "base"
    language: Optional[str] = None

class TranscribeResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: Optional[List[Segment]] = None

class SynthesizeRequest(BaseModel):
    text: str
    model: str = "kokoro"
    voice: str = "default"
    speed: float = 1.0
    response_format: str = "wav"  # wav, mp3, b64_json

# Endpoints
@app.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    request: TranscribeRequest = TranscribeRequest()
):
    audio = await load_audio(file)
    result = await asyncio.to_thread(
        audio_engine.transcribe, audio, **request.dict()
    )
    return result

@app.post("/api/synthesize")
async def synthesize(request: SynthesizeRequest):
    result = await asyncio.to_thread(
        audio_engine.synthesize, request.text, **request.dict()
    )
    if request.response_format == "b64_json":
        return {"audio": base64.b64encode(result.audio).decode()}
    return Response(content=result.to_wav(), media_type="audio/wav")

@app.post("/api/clone-voice")
async def clone_voice(
    reference: UploadFile = File(...),
    text: str = Form(...)
):
    """Clone voice from reference audio."""
    pass

@app.websocket("/api/stream")
async def stream_transcribe(websocket: WebSocket):
    """Real-time streaming transcription."""
    streamer = StreamingTranscriber()
    await streamer.handle_stream(websocket)
```

---

## Phase 6: Testing & Documentation

### 6.1 Test Structure

```
tests/
├── unit/
│   ├── test_stt_strategies.py
│   ├── test_tts_strategies.py
│   ├── test_model_registry.py
│   └── test_audio_engine.py
├── integration/
│   ├── test_cli_commands.py
│   ├── test_api_endpoints.py
│   └── test_mcp_server.py
└── fixtures/
    └── audio_samples/
```

### 6.2 Benchmarking Tool

```python
# cli/commands/benchmark.py
@click.command()
@click.option("--models", "-m", multiple=True)
@click.option("--audio", "-a", required=True)
def benchmark(models, audio):
    """Benchmark STT models for speed and accuracy."""
    results = []
    for model in models:
        start = time.time()
        result = audio_engine.transcribe(audio, model=model)
        elapsed = time.time() - start
        results.append({
            "model": model,
            "time": elapsed,
            "rtf": elapsed / audio_duration,  # Real-time factor
            "text": result.text
        })

    # Display comparison table
    display_benchmark_results(results)
```

---

## Implementation Order

### Sprint 1: Foundation (Week 1-2)
- [ ] Create Strategy base classes (STT/TTS)
- [ ] Migrate existing engines to strategies (whisper, faster-whisper, whisper-cpp, kokoro, xtts)
- [ ] Implement AudioEngine facade
- [ ] Enhanced ModelRegistry with external config support
- [ ] Add TranscriptionResult and AudioResult dataclasses

### Sprint 2: CLI Restructure (Week 3)
- [ ] Break cli.py into command modules
- [ ] Implement OllamaStyleProgress
- [ ] Add hardware detection and recommendations
- [ ] Update entry points
- [ ] Add `kin audio benchmark` command

### Sprint 3: New STT Models (Week 4-5)
**English/General:**
- [ ] Add Moonshine strategy (ultra-fast)
- [ ] Add Parakeet TDT strategy (NVIDIA)
- [ ] Add Canary strategy (NVIDIA)
- [ ] Add Distil-Whisper strategy

**Chinese/Multilingual:**
- [ ] Add SenseVoice strategy (Alibaba - 15x faster, emotion)
- [ ] Add Paraformer strategy (Alibaba - Chinese)
- [ ] Add FunASR strategy (toolkit with VAD/punctuation)
- [ ] Add FireRedASR strategy (dialects)

### Sprint 4: New TTS Models (Week 6-7)
**English/General:**
- [ ] Add F5-TTS strategy (voice cloning)
- [ ] Add Parler-TTS strategy (text-described voices)
- [ ] Add MeloTTS strategy (multilingual)
- [ ] Add OuteTTS strategy (lightweight)

**Chinese:**
- [ ] Add CosyVoice 2 strategy (Alibaba SOTA)
- [ ] Add ChatTTS strategy (conversational)
- [ ] Add GPT-SoVITS v3 strategy (voice cloning)
- [ ] Add EmotiVoice strategy (emotional)

### Sprint 5: New Features (Week 8-9)
- [ ] MCP server implementation
- [ ] Voice cloning API (`/api/clone-voice`)
- [ ] WebSocket streaming transcription
- [ ] Cross-lingual TTS support
- [ ] Emotion control for TTS
- [ ] Benchmarking tool

### Sprint 6: Polish & Release (Week 10)
- [ ] Comprehensive tests for all strategies
- [ ] API documentation (OpenAPI)
- [ ] Update README with new models
- [ ] Chinese documentation (README_CN.md)
- [ ] Version bump to 2.0.0
- [ ] PyPI release

---

## New Dependencies

### Required for New Models

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
# Chinese models
chinese = [
    "funasr>=1.0.0",           # SenseVoice, Paraformer, FunASR
    "modelscope>=1.9.0",       # Model hub for Chinese models
    "cosyvoice>=0.1.0",        # CosyVoice 2
    "ChatTTS>=0.1.0",          # ChatTTS
    "GPT-SoVITS>=3.0.0",       # GPT-SoVITS v3
]

# Voice cloning
cloning = [
    "f5-tts>=0.1.0",           # F5-TTS
    "cosyvoice>=0.1.0",        # CosyVoice 2
    "GPT-SoVITS>=3.0.0",       # GPT-SoVITS v3
]

# NVIDIA models
nvidia = [
    "nemo-toolkit>=1.20.0",    # Parakeet, Canary
]

# Fast models
fast = [
    "moonshine>=0.1.0",        # Moonshine
    "funasr>=1.0.0",           # SenseVoice
]

# All new models
all-models = [
    "localkin-service-audio[chinese,cloning,nvidia,fast]",
    "parler-tts>=0.1.0",       # Parler TTS
    "melotts>=0.1.0",          # MeloTTS
    "emotivoice>=0.1.0",       # EmotiVoice
]
```

### Installation Examples

```bash
# Install with Chinese model support
pip install localkin-service-audio[chinese]

# Install with voice cloning
pip install localkin-service-audio[cloning]

# Install everything
pip install localkin-service-audio[all-models]

# Using uv
uv pip install localkin-service-audio[chinese,cloning]
```

---

## Breaking Changes (v1.x → v2.0)

1. **CLI argument changes:**
   - `--engine` flag required for explicit engine selection
   - Model names standardized (e.g., `whisper-cpp:tiny` format)

2. **API changes:**
   - New endpoint structure under `/api/`
   - Response format changes for consistency

3. **Configuration:**
   - New `~/.localkin-service-audio/config.yaml` format
   - Old cache location still supported

---

## File Summary

| File | Changes |
|------|---------|
| `core/audio_processing/engine.py` | NEW - Facade for audio operations |
| `core/audio_processing/stt/base.py` | NEW - STT strategy ABC |
| `core/audio_processing/stt/*.py` | NEW - Individual STT strategies |
| `core/audio_processing/tts/base.py` | NEW - TTS strategy ABC |
| `core/audio_processing/tts/*.py` | NEW - Individual TTS strategies |
| `core/config/model_registry.py` | NEW - Enhanced model registry |
| `cli/main.py` | REFACTOR - Slim entry point |
| `cli/commands/*.py` | NEW - Command modules |
| `mcp/server.py` | NEW - MCP integration |
| `api/server.py` | ENHANCE - New endpoints |
| `api/websocket.py` | NEW - Streaming support |
