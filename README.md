# LocalKin Service Audio 🎵

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/⚡-uv-4c1d95)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LocalKin Service Audio** simplifies local deployment of **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** models. An intuitive **local audio** tool inspired by **Ollama's** simplicity - perfect for **local audio processing** workflows with both CLI and modern web interface support.

## ✨ Features

- **🚀 Fast Startup**: Instant application launch with lazy loading architecture
- **⚡ Faster-Whisper Integration**: Up to 4x faster transcription with CTranslate2 optimization
- **🎯 Multiple STT Engines**: OpenAI Whisper, faster-whisper, and Hugging Face models
- **🔊 Multiple TTS Engines**: Native OS TTS, SpeechT5, Bark, and Coqui XTTS v2
- **🌐 REST API Server**: Run models as API servers with automatic endpoints
- **💻 Modern Web Interface**: Beautiful, responsive web UI for easy audio processing
- **📦 Smart Model Management**: Auto-pull models when needed, intelligent caching
- **💾 Persistent Cache**: Local model storage with size tracking and cleanup
- **📊 Real-Time Status**: Live model status tracking with emoji indicators
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration support
- **🔧 Modular Architecture**: Clean, maintainable codebase with separated concerns

## 🚀 Quick Start

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio
uv sync

# Verify installation
kin --version
```

### Basic Usage

```bash
# List available models
kin audio models

# Transcribe audio file
kin audio transcribe audio.wav

# Generate speech
kin audio tts "Hello, world!"

# Start web interface
kin web

# Run model as API server
kin audio run faster-whisper-tiny --port 8000
```

## 🎯 Supported Models

### STT Models
- **faster-whisper-tiny/base/small/medium/large** - Up to 4x faster transcription
- **whisper-tiny/base/large-v2** - Original OpenAI Whisper models
- **speecht5** - Microsoft speech recognition

### TTS Models
- **pyttsx3** - Native OS TTS (instant)
- **speecht5-tts** - Microsoft neural TTS (fast)
- **kokoro-82m** - High-quality neural TTS
- **bark-small** - Suno Bark TTS
- **xtts-v2** - Coqui multilingual voice cloning

### Model Status Indicators
- 📦 **Local Library**: Built-in models (no download needed)
- ✅ **Pulled**: Downloaded and cached locally
- ⬇️ **Not Pulled**: Available for download when needed

```bash
$ kin audio models

🎵 LocalKin Service Audio - Local STT & TTS Model Manager
============================================================

MODEL                     TYPE   STATUS             SOURCE          DESCRIPTION
------------------------------------------------------------------------------------------
whisper                   stt    📦 Local Library    openai-whisper  OpenAI Whisper with auto faster-whisper
faster-whisper-tiny       stt    📦 Local Library    faster-whisper  Fast Whisper Tiny - 4x faster, smallest size
speecht5                  stt    ⬇️  Not Pulled     huggingface     Microsoft SpeechT5 STT
------------------------------------------------------------------------------------------
native                    tts    📦 Local Library    pyttsx3         Native macOS TTS via pyttsx3
speecht5-tts              tts    ⬇️  Not Pulled     huggingface     Microsoft SpeechT5 TTS
kokoro-82m                tts    ✅ Pulled           huggingface     HexGrad Kokoro-82M - high-quality neural TTS
xtts-v2                   tts    ⬇️  Not Pulled     huggingface     Coqui XTTS v2 - multilingual voice cloning

📊 Total models: 8
```

## 🎛️ Usage Examples

### Web Interface
```bash
# Launch modern web UI
kin web

# Open http://localhost:8080 in your browser
```

### API Servers
```bash
# Start STT server
kin audio run faster-whisper-tiny --port 8000

# Start TTS server
kin audio run kokoro-82m --port 8001

# Use the API
curl -X POST "http://localhost:8000/transcribe" -F "file=@audio.wav"
curl -X POST "http://localhost:8001/synthesize" -H "Content-Type: application/json" -d '{"text": "Hello!"}'
```

### Cache Management
```bash
# Check cache status
kin audio cache info

# Clear specific model
kin audio cache clear model-name

# Clear all cache
kin audio cache clear
```

## 🔧 Advanced Features

### Optional Dependencies
```bash
# Install with web interface
uv sync --extra web

# Install with advanced TTS models (XTTS)
uv sync --extra tts

# Install with GPU support
uv sync --extra gpu

# Install everything
uv sync --extra web --extra tts --extra gpu
```

### Performance Tuning
- **GPU**: Automatic CUDA/MPS detection
- **Memory**: Efficient model loading and caching
- **Batch**: Parallel processing support

### Troubleshooting
- **Model Download**: Check internet and clear cache
- **Memory Issues**: Use smaller models or CPU mode
- **Audio Format**: Convert to WAV/MP3/FLAC

## 📋 API Reference

### STT Endpoint
```bash
POST /transcribe
Content-Type: multipart/form-data

file: <audio_file>
```

### TTS Endpoint
```bash
POST /synthesize
Content-Type: application/json

{"text": "Hello, world!", "speaker": "optional"}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for Whisper
- SYSTRAN for faster-whisper
- Coqui TTS for XTTS
- Hugging Face for model hosting

---

**Ready to get started?** `kin web` for the web interface, or `kin audio models` to explore available models! 🎵✨
