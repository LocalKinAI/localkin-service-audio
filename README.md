# LocalKin Service Audio 🎵

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/⚡-uv-4c1d95)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Complete Voice AI Platform: STT, TTS & LLM Integration

**LocalKin Service Audio** is a complete **voice AI platform** featuring **Speech-to-Text (STT)**, **Text-to-Speech (TTS)**, and **Large Language Model (LLM) integration**. An intuitive **local audio** tool inspired by **Ollama's** simplicity - perfect for **voice-based conversational AI** with both CLI and modern web interface support.

## ✨ Key Features

- **🚀 Fast Startup**: Instant application launch with lazy loading architecture
- **⚡ Maximum Performance**: whisper.cpp integration for up to 50x faster transcription
- **🎯 Multiple STT Engines**: OpenAI Whisper, faster-whisper with VAD, whisper.cpp (C++), and Hugging Face models
- **🔊 Multiple TTS Engines**: Native OS TTS, SpeechT5, Bark, Kokoro, and XTTS models
- **🌐 REST API Server**: Run models as API servers with automatic endpoints
- **💻 Modern Web Interface**: Beautiful, responsive web UI with file upload, voice selection, and dynamic model discovery
- **🤖 LLM Integration**: Voice-based conversational AI with Ollama models for intelligent responses
- **🎭 Voice Selection**: Multiple voice options for TTS models (Kokoro, XTTS, SpeechT5)
- **📄 File Upload Support**: Upload text files (.txt, .md, .rtf) for TTS synthesis
- **🔍 Dynamic API Discovery**: Automatically finds and uses running API servers
- **📦 Smart Model Management**: Auto-pull models when needed, intelligent caching
- **💾 Persistent Cache**: Local model storage with size tracking and cleanup
- **🔄 Auto-Pull**: Models automatically download when running if not cached
- **📊 Real-Time Status**: Live model status tracking with emoji indicators
- **🔍 Process Monitoring**: `kin audio ps` shows all running servers and their status
- **📈 Model Transparency**: STT/TTS commands display detailed model information and statistics
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration support
- **🎨 Professional Results**: High-quality audio processing with fine-tuned control
- **🌐 CLI & Web**: Both command-line interface and modern web interface
- **🔧 Modular Architecture**: Clean, maintainable codebase with separated concerns

## 🚀 Quick Start

```bash
# Install uv if you haven't already (highly recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install LocalKin Service Audio
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio
uv sync

# Start using it!
kin --help
```

### Basic Usage

```bash
# Check version and help
kin --version
kin --help

# List all available models with status
kin audio models

# Transcribe audio files
kin audio transcribe audio.wav                    # Auto-select best engine
kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny  # Ultra-fast
kin audio transcribe audio.wav --engine faster --vad                   # With VAD

# Real-time listening with TTS
kin audio listen --engine whisper-cpp --tts --tts-model native

# Voice AI with streaming LLM responses
kin audio listen --llm ollama --tts --stream

# Synthesize speech
kin audio tts "Hello world" --model kokoro-82m

# Start web interface
kin web
```

## 🎯 Supported Models

### STT Models - Speech-to-Text

#### 🚀 Ultra-Fast whisper.cpp Models (Recommended)
```bash
# Download models
python scripts/download_whisper_cpp_models.py --list
python scripts/download_whisper_cpp_models.py tiny base small

# Usage
kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny  # 50x faster!
kin audio listen --engine whisper-cpp --model_size base               # Real-time
```

**Performance:** Up to 50x faster than OpenAI Whisper with low memory usage.

#### ⚡ Fast Whisper Models (Balanced)
```bash
# Usage (auto-downloads on first use)
kin audio transcribe audio.wav --engine faster --vad  # 4x-32x faster with VAD
kin audio transcribe audio.wav --model faster-whisper-tiny
```

#### 🏠 Local Whisper Models (Compatible)
```bash
# Built-in, no download needed
kin audio transcribe audio.wav --engine openai --model_size base
kin audio transcribe audio.wav  # Auto-selects best available
```

### TTS Models - Text-to-Speech

#### 🚀 API Server Models (Recommended for Production)
```bash
# Start API server (loads once, instant responses)
kin audio run kokoro-82m --port 8001      # High quality, 320MB
kin audio run speecht5-tts --port 8002    # Fast, 130MB
kin audio run xtts-v2 --port 8003         # Voice cloning, 1.8GB

# Use API
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output speech.wav
```

#### 💻 CLI Models (Development/Testing)
```bash
# Direct CLI usage (loads on-demand)
kin audio tts "Hello world" --model kokoro-82m
kin audio tts "Hello world" --model native        # Fastest, built-in
```

### 🤖 LLM Integration

#### Voice-Based Conversational AI
```bash
# Requires Ollama running
kin audio listen --llm ollama --tts                    # Full voice AI
kin audio listen --llm ollama --llm-model qwen3:14b    # Custom model
kin audio listen --llm ollama --tts --stream           # Streaming responses
kin audio listen --engine whisper-cpp --model_size small --tts --tts-model kokoro-82m --vad --llm ollama --llm-model deepseek-r1:14b --stream
```

**Streaming Mode**: Add `--stream` for real-time LLM responses that speak as they generate, creating more natural conversational flow.

## 📦 Installation & Setup

### Prerequisites
- **Python 3.10+** (required for optimal performance)
- **uv** (fast Python package installer - highly recommended)
- **Ollama** (optional, for LLM integration)
- **FFmpeg** (for audio processing)

### Quick Install with uv
```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio
uv sync
```

### whisper.cpp Setup (Optional, for Maximum Performance)
```bash
# Build whisper.cpp (requires CMake, Make, C++ compiler)
./scripts/build_whisper_cpp.sh

# Download models
python scripts/download_whisper_cpp_models.py tiny base small
```

### Ollama Setup (Optional, for LLM Integration)
```bash
# Install Ollama
brew install ollama  # macOS
# OR visit https://ollama.ai for other platforms

# Start Ollama service
ollama serve

# Pull recommended models
ollama pull qwen3:14b
```

## ⚡ Performance & Benchmarks

### Engine Comparison

| Engine | Speed | Memory | VAD | GPU Support | Best For |
|--------|-------|--------|-----|-------------|----------|
| **whisper.cpp** | **50x** | **Low** | ❌ | ❌ | **Maximum Performance** |
| **faster-whisper** | 4x-32x | Medium | ✅ | ✅ | Balanced speed/quality |
| **OpenAI Whisper** | 1x | High | ❌ | ✅ | Compatibility |

### Hardware Recommendations

- **Basic Usage**: 8GB RAM, any CPU
- **High-Quality Models**: 16GB+ RAM, GPU recommended
- **whisper.cpp**: Works on any hardware, best performance
- **Real-time Applications**: Use whisper.cpp for lowest latency

## 🌐 REST API Reference

### Running API Servers
```bash
# STT API server
kin audio run whisper-cpp-tiny --port 8000

# TTS API servers
kin audio run kokoro-82m --port 8001
kin audio run speecht5-tts --port 8002
```

### API Endpoints

#### STT Endpoint
```bash
# Transcribe audio
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@audio.wav" \
     -F "language=en"
```

#### TTS Endpoints
```bash
# Synthesize speech
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output speech.wav
```

## 💾 Cache Management

### Cache Commands
```bash
# Check cache status
kin audio cache info

# Clear specific model
kin audio cache clear whisper-tiny-hf

# Clear all cached models
kin audio cache clear
```

### Auto-Pull Behavior
Models are automatically downloaded when first used. No manual intervention required!

## 🔧 Troubleshooting

### Common Issues

#### "Model not found" Error
```bash
# Check available models
kin audio models

# For Ollama models, ensure Ollama is running
ollama serve

# Pull the model first
kin audio pull whisper-base
```

#### Audio File Issues
```bash
# Ensure audio files are in supported formats: WAV, MP3, FLAC, OGG
# For best results, use 16-bit WAV at 16kHz sample rate
```

#### whisper.cpp Library Issues
```bash
# If you get library loading errors, rebuild whisper.cpp
./scripts/build_whisper_cpp.sh
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio

# Install in development mode
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

### Adding New Models
See the [model configuration guide](docs/model-configuration.md) for details on adding new STT/TTS models.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for the Whisper model
- **ggerganov** for whisper.cpp
- **SYSTRAN** for faster-whisper
- **Hugging Face** for model hosting
- **Ollama** for the inspiration and local AI ecosystem

---

**🎉 Ready to get started with local audio AI? Install LocalKin Service Audio and choose your preferred interface!**
