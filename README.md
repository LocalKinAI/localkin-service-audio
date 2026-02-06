# LocalKin Service Audio

[![PyPI version](https://badge.fury.io/py/localkin-service-audio.svg)](https://pypi.org/project/localkin-service-audio/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Local Voice AI Platform** - Speech-to-Text and Text-to-Speech with Chinese language support, voice cloning, and Claude integration via MCP.

## Features

- **Multiple STT Engines**: Whisper, faster-whisper, whisper.cpp, SenseVoice, Paraformer
- **Multiple TTS Engines**: Kokoro, CosyVoice, ChatTTS, F5-TTS, native OS
- **Chinese Language Support**: Optimized models for Mandarin, Cantonese, and mixed Chinese-English
- **Voice Cloning**: Zero-shot voice cloning with F5-TTS and CosyVoice
- **MCP Integration**: Use with Claude Code and Claude Desktop
- **WebSocket Streaming**: Real-time transcription and synthesis
- **REST API**: FastAPI-based server with OpenAPI docs

## Quick Start

```bash
# Install (uv recommended)
uv pip install localkin-service-audio

# Get model recommendations for your hardware
kin audio recommend

# View configuration
kin audio config

# Transcribe audio
kin audio transcribe audio.wav

# Text-to-speech
kin audio tts "Hello world"

# Real-time listening (microphone)
kin audio listen

# Voice AI conversation
kin audio listen --llm ollama --tts --stream

# List available models
kin audio models

# Start API server
kin audio serve --port 8000

# Start web interface
kin web
```

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is 10-100x faster than pip with better dependency resolution.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install LocalKin Audio
uv pip install localkin-service-audio

# Or install from source (for development)
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio
uv sync
```

### Using pip

```bash
pip install localkin-service-audio
```

### Optional Dependencies

```bash
# Chinese language models
pip install localkin-service-audio[chinese]

# Voice cloning models
pip install localkin-service-audio[cloning]

# MCP server for Claude
pip install localkin-service-audio[mcp]

# All features
pip install localkin-service-audio[all-new]
```

## CLI Usage

### Speech-to-Text

```bash
# Basic transcription (auto-selects best model)
kin audio transcribe audio.wav

# Specify model
kin audio transcribe audio.wav --model whisper-cpp:base
kin audio transcribe audio.wav --model faster-whisper:large-v3
kin audio transcribe audio.wav --model sensevoice:small  # Chinese

# With language hint
kin audio transcribe audio.wav --language zh

# Output formats
kin audio transcribe audio.wav --format json
kin audio transcribe audio.wav --format srt --timestamps
```

### Text-to-Speech

```bash
# Basic synthesis (uses Kokoro with af_heart voice)
kin audio tts "Hello world"

# List all available voices
kin audio tts "" --model kokoro --list-voices

# American English voices
kin audio tts "Hello world" --voice af_bella       # Bella (Female)
kin audio tts "Hello world" --voice am_adam         # Adam (Male)
kin audio tts "Hello world" --voice af_nova         # Nova (Female)

# British English voices
kin audio tts "Good morning" --voice bf_emma        # Emma (British Female)
kin audio tts "Good morning" --voice bm_george      # George (British Male)

# Chinese (Mandarin) voices
kin audio tts "你好世界" --voice zf_xiaoxiao         # Xiaoxiao (Chinese Female)
kin audio tts "今天天气真好" --voice zm_yunyang       # Yunyang (Chinese Male)

# Japanese voices
kin audio tts "こんにちは" --voice jf_alpha           # Alpha (Japanese Female)
kin audio tts "ありがとう" --voice jm_kumo            # Kumo (Japanese Male)

# French, Spanish, Italian, Hindi, Portuguese
kin audio tts "Bonjour le monde" --voice ff_siwis   # French
kin audio tts "Hola mundo" --voice ef_dora           # Spanish
kin audio tts "Ciao mondo" --voice if_sara           # Italian
kin audio tts "नमस्ते" --voice hf_alpha              # Hindi
kin audio tts "Olá mundo" --voice pf_dora            # Portuguese

# Adjust speech speed (0.5 = slow, 2.0 = fast)
kin audio tts "Hello world" --speed 0.8
kin audio tts "Hello world" --speed 1.5

# Save to file
kin audio tts "Hello world" --output speech.wav

# Save without auto-playing
kin audio tts "Hello world" --output speech.wav --no-play

# CosyVoice for Chinese (voice cloning capable)
kin audio tts "你好世界" --model cosyvoice:300m --voice 中文女
```

### Real-time Listening

```bash
# Basic real-time transcription
kin audio listen

# With TTS echo
kin audio listen --tts --tts-model kokoro

# Voice AI with LLM (requires Ollama)
kin audio listen --llm ollama --tts --stream

# Custom models
kin audio listen --model sensevoice:small --language zh --tts --tts-model cosyvoice:300m

# Adjust silence detection
kin audio listen --silence-threshold 0.02 --silence-duration 2.0
```

### Model Management

```bash
# List all models
kin audio models

# List only STT models
kin audio models --type stt

# List only TTS models
kin audio models --type tts

# Pull a model
kin audio pull whisper-cpp:base

# Remove a model
kin audio rm whisper-cpp:base

# Add a model from a template
kin audio add-model --template whisper_stt --name my-whisper

# Add a model from HuggingFace
kin audio add-model --repo openai/whisper-medium --name whisper-med --type stt

# List available model templates
kin audio list-templates
```

### Model Recommendations

```bash
# Get hardware-aware model recommendations
kin audio recommend

# With detailed hardware info
kin audio recommend --verbose
```

The recommend command detects your hardware (GPU, RAM, CPU) and suggests optimal STT/TTS models for your system.

### Configuration

```bash
# View configuration overview
kin audio config

# Show configuration file paths
kin audio config --path

# Show all registered models
kin audio config --models

# Initialize config directory with sample config
kin audio config --init

# Change settings
kin audio config set default_tts_model kokoro
kin audio config set default_stt_model faster-whisper:large-v3
kin audio config set api_port 9000
kin audio config set default_device cuda
```

Configuration files are stored in `$LOCALKIN_HOME/` (default: `~/.localkin-service-audio/`).

Set `LOCALKIN_HOME` to relocate all data (cache, config, models) to another disk:

```bash
export LOCALKIN_HOME="/path/to/large/disk/.localkin-service-audio"
```

### System Status & Diagnostics

```bash
# Check system status (libraries, registry, cache)
kin audio status

# Show cache info
kin audio cache info

# Clear cache for a specific model
kin audio cache clear whisper-large

# Clear all cached models
kin audio cache clear

# Show running LocalKin Audio servers
kin audio ps
```

### API Server

```bash
# Start REST API server
kin audio serve --port 8000

# Start web interface
kin web --port 5000
```

## Supported Models

### STT Models

| Model | Engine | Languages | Features |
|-------|--------|-----------|----------|
| `whisper:base` | OpenAI Whisper | Multilingual | Standard accuracy |
| `whisper-cpp:base` | whisper.cpp | Multilingual | Fast CPU inference |
| `faster-whisper:large-v3` | CTranslate2 | Multilingual | 4x faster, GPU |
| `sensevoice:small` | FunASR | zh, en, ja, ko | Emotion detection |
| `paraformer:zh` | FunASR | Chinese | Fast Chinese ASR |

### TTS Models

| Model | Engine | Languages | Voices | Features |
|-------|--------|-----------|--------|----------|
| `native` | pyttsx3 | System | OS default | No download |
| `kokoro` | Kokoro | en, es, fr, hi, it, ja, pt, zh | 54 voices | High quality, multilingual |
| `cosyvoice:300m` | CosyVoice | zh, en, ja, ko, yue | Multiple | Voice cloning |
| `chattts` | ChatTTS | zh, en | Random seed | Conversational |
| `f5-tts` | F5-TTS | en, zh | Reference audio | Zero-shot cloning |

### Kokoro Voice Reference

Kokoro supports 54 voices across 9 languages. Voice IDs follow the pattern `{lang}{gender}_{name}`:

| Prefix | Language | Example Voices |
|--------|----------|----------------|
| `af_` | American English (Female) | `af_heart`, `af_bella`, `af_nova`, `af_sarah`, `af_sky` |
| `am_` | American English (Male) | `am_adam`, `am_michael`, `am_echo`, `am_puck` |
| `bf_` | British English (Female) | `bf_emma`, `bf_alice`, `bf_lily`, `bf_isabella` |
| `bm_` | British English (Male) | `bm_george`, `bm_lewis`, `bm_daniel`, `bm_fable` |
| `zf_` | Chinese Mandarin (Female) | `zf_xiaoxiao`, `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoyi` |
| `zm_` | Chinese Mandarin (Male) | `zm_yunyang`, `zm_yunxi`, `zm_yunjian`, `zm_yunxia` |
| `jf_` | Japanese (Female) | `jf_alpha`, `jf_nezumi`, `jf_gongitsune`, `jf_tebukuro` |
| `jm_` | Japanese (Male) | `jm_kumo` |
| `ff_` | French (Female) | `ff_siwis` |
| `ef_` | Spanish (Female) | `ef_dora` |
| `em_` | Spanish (Male) | `em_alex` |
| `hf_` | Hindi (Female) | `hf_alpha`, `hf_beta` |
| `hm_` | Hindi (Male) | `hm_omega`, `hm_psi` |
| `if_` | Italian (Female) | `if_sara` |
| `im_` | Italian (Male) | `im_nicola` |
| `pf_` | Portuguese (Female) | `pf_dora` |
| `pm_` | Portuguese (Male) | `pm_alex` |

## Python API

```python
from localkin_service_audio import AudioEngine, transcribe, synthesize

# Quick functions
result = transcribe("audio.wav", model="whisper-cpp:base")
print(result.text)

audio = synthesize("Hello world", model="kokoro")
audio.save("output.wav")

# Full engine control
engine = AudioEngine()

# Load and use STT
engine.load_stt("whisper-cpp:base")
result = engine.transcribe("audio.wav", language="en")
print(f"Text: {result.text}")
print(f"Language: {result.language}")

# Load and use TTS - English
engine.load_tts("kokoro")
audio = engine.synthesize("Hello world", voice="af_heart")
audio.save("english.wav")

# TTS - Chinese (auto-selects Chinese pipeline)
audio = engine.synthesize("你好世界", voice="zf_xiaoxiao")
audio.save("chinese.wav")

# TTS - Japanese
audio = engine.synthesize("こんにちは世界", voice="jf_alpha")
audio.save("japanese.wav")

# TTS - with speed control
audio = engine.synthesize("Hello", voice="am_adam", speed=0.8)
audio.save("slow.wav")

# List available voices
voices = engine.list_voices()
for v in voices:
    print(f"{v.id}: {v.name} ({v.language}, {v.gender})")

# Voice cloning (with supported models)
engine.load_tts("f5-tts")
audio = engine.clone_voice(
    reference_audio="reference.wav",
    text="Text to speak in cloned voice"
)
```

## MCP Integration

Use LocalKin Audio with Claude Code or Claude Desktop:

```bash
# Start MCP server
kin mcp
```

Add to Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "localkin-audio": {
      "command": "kin",
      "args": ["mcp"]
    }
  }
}
```

Available MCP tools:
- `transcribe_audio` - Transcribe audio files
- `synthesize_speech` - Generate speech from text
- `clone_voice` - Clone voice from reference audio
- `list_models` - List available models
- `list_voices` - List available voices

## REST API

Start the server:

```bash
kin audio serve --port 8000
```

### Endpoints

**POST /transcribe** - Transcribe audio
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "model=whisper-cpp:base" \
  -F "language=en"
```

**POST /synthesize** - Synthesize speech
```bash
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "kokoro", "voice": "af_bella"}' \
  --output speech.wav
```

**GET /models** - List models
```bash
curl "http://localhost:8000/models"
```

**WebSocket /stream** - Real-time transcription
```javascript
const ws = new WebSocket("ws://localhost:8000/stream");
ws.send(audioChunk);
ws.onmessage = (e) => console.log(JSON.parse(e.data).text);
```

## Configuration

### Environment Variables

```bash
# Base directory for all data (cache, config, models)
export LOCALKIN_HOME="/Volumes/Data/.localkin-service-audio"

# Override individual directories
export LOCALKIN_CACHE_DIR="/tmp/my-cache"
export LOCALKIN_CONFIG_DIR="/path/to/config"
export LOCALKIN_MODELS_DIR="/path/to/models"

# Default engine settings
export LOCALKIN_DEFAULT_STT="faster-whisper:large-v3"
export LOCALKIN_DEFAULT_TTS="kokoro"
export LOCALKIN_DEVICE=cuda  # or cpu, mps, auto

# API server
export LOCALKIN_API_HOST="127.0.0.1"
export LOCALKIN_API_PORT="8000"
```

### Custom Models

Create `$LOCALKIN_HOME/models.json` (default: `~/.localkin-service-audio/models.json`):

```json
{
  "models": {
    "my-custom-model": {
      "type": "stt",
      "engine": "whisper",
      "model_size": "base",
      "languages": ["en", "zh"],
      "description": "My custom model"
    }
  }
}
```

## Architecture

LocalKin Audio v2.0 uses a modular architecture:

- **Strategy Pattern**: Pluggable STT/TTS engines
- **Facade Pattern**: AudioEngine provides unified interface
- **Registry Pattern**: Centralized model configuration
- **Singleton Pattern**: Shared engine instance

```
localkin_service_audio/
├── core/
│   ├── audio_processing/
│   │   ├── engine.py          # AudioEngine facade
│   │   ├── stt/               # STT strategies
│   │   │   ├── base.py
│   │   │   ├── whisper_strategy.py
│   │   │   ├── sensevoice_strategy.py
│   │   │   └── ...
│   │   └── tts/               # TTS strategies
│   │       ├── base.py
│   │       ├── kokoro_strategy.py
│   │       ├── cosyvoice_strategy.py
│   │       └── ...
│   ├── config/
│   │   └── model_registry.py  # Model registry
│   └── types.py               # Core dataclasses
├── cli/                       # Click CLI
├── api/                       # FastAPI server
├── mcp/                       # MCP server
└── ui/                        # Web interface
```

## Development

```bash
# Clone repository
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
black --check .
```

## Troubleshooting

### Model Loading Errors

```bash
# Check model is registered
kin audio models

# Pull the model
kin audio pull whisper-cpp:base

# Check system info
kin info --verbose
```

### CUDA/GPU Issues

```bash
# Force CPU
kin audio transcribe audio.wav --device cpu

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Chinese Model Dependencies

```bash
# Install FunASR for Chinese models
pip install funasr modelscope

# Then use Chinese models
kin audio transcribe audio.wav --model sensevoice:small
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
