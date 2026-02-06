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
# Basic synthesis
kin audio tts "Hello world"

# Specify model and voice
kin audio tts "Hello world" --model kokoro --voice af_bella
kin audio tts "你好世界" --model cosyvoice:300m --voice 中文女

# Save to file
kin audio tts "Hello world" --output speech.wav
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
```

Configuration files are stored in `~/.localkin-service-audio/`.

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

| Model | Engine | Languages | Features |
|-------|--------|-----------|----------|
| `native` | pyttsx3 | System | No download |
| `kokoro` | Kokoro | English | High quality |
| `cosyvoice:300m` | CosyVoice | zh, en, ja, ko | Voice cloning |
| `chattts` | ChatTTS | zh, en | Conversational |
| `f5-tts` | F5-TTS | en, zh | Zero-shot cloning |

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

# Load and use TTS
engine.load_tts("kokoro")
audio = engine.synthesize("Hello world", voice="af_bella")
audio.save("output.wav")

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
# Custom model config file
export LOCALKIN_MODEL_CONFIG=~/.localkin/models.yaml

# Default device
export LOCALKIN_DEVICE=cuda  # or cpu, mps
```

### Custom Models

Create `~/.localkin-service-audio/models.json`:

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
