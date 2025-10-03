# LocalKin Service Audio 🎵

[![PyPI version](https://badge.fury.io/py/localkin-service-audio.svg)](https://pypi.org/project/localkin-service-audio/)
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

**📋 New to this?** If you're setting up on a clean machine or want a complete setup guide with Python, pyenv, virtual environments, whisper.cpp, and more, see the [Complete Installation & Setup Guide](#-installation--setup) below.

### Recommended: Install with uv (Best for Kokoro TTS)

Using `uv` ensures you have proper Python environment with LZMA support for Kokoro TTS:

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install LocalKin Service Audio
uv pip install localkin-service-audio

# Start using it!
kin --help

# Try Kokoro TTS (requires LZMA support)
kin audio run kokoro-82m --port 8001
```

### Alternative: Install with pip

If you prefer traditional pip (may have LZMA issues with pyenv):

```bash
# Install from PyPI
pip install localkin-service-audio

# or upgrade
pip uninstall localkin-service-audio
pip install localkin-service-audio --upgrade --no-cache-dir

# Start using it!
kin --help

# If you get LZMA errors with Kokoro, see troubleshooting below or use uv
```

**💡 Pro Tips:** 
- If you encounter "Could not import module 'pipeline'" errors with Kokoro TTS, use the `uv` installation method or see [Troubleshooting](#kokoro-tts-could-not-import-module-pipeline-error)
- For clean machine setup, see the [Complete Installation Guide](#-installation--setup) below
- Need whisper.cpp or Ollama? See [Step 5](#step-5-install-whispercpp-optional---for-faster-stt) and [Step 7](#step-7-install-ollama-optional---for-voice-ai-features)

### Basic Usage

```bash
# Check version and help
kin --version
kin --help

# List all available models with status
kin audio models

# Get the included sample audio file for testing
python -c "import localkin_service_audio as lsa; print(lsa.get_sample_audio_path())"

# Transcribe audio files (use sample.wav for testing)
kin audio transcribe audio.wav                    # Auto-select best engine
kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny  # Ultra-fast
kin audio transcribe audio.wav --engine faster --vad                   # With VAD

# Or use the included sample audio:
python -c "import localkin_service_audio as lsa; from localkin_service_audio import transcribe_audio; print(transcribe_audio(lsa.get_sample_audio_path()))"

# Real-time listening with TTS
kin audio listen --engine whisper-cpp --tts --tts-model native

# Voice AI with streaming LLM responses
kin audio listen --llm ollama --tts --stream

# Synthesize speech (native/pyttsx3)
kin audio tts "Hello world" --model native

# For Kokoro TTS, use the API server (more reliable)
kin audio run kokoro-82m --port 8001  # Start server
# Then use API: curl -X POST "http://localhost:8001/synthesize" -H "Content-Type: application/json" -d '{"text":"Hello world"}'

# Start web interface
kin web
```

**📦 Sample Audio Included:** A sample audio file is included in the package for immediate testing. Access it with `localkin_service_audio.get_sample_audio_path()`.

**💡 Note:** Kokoro TTS works best via the API server (`kin audio run kokoro-82m`). Direct CLI synthesis (`kin audio tts`) is best for native/pyttsx3 models.

## 🎯 Supported Models

### STT Models - Speech-to-Text

#### 🚀 Ultra-Fast whisper.cpp Models (Recommended)
```bash
# Automatic installation - no manual setup needed!
# Models download automatically on first use

# Usage - works immediately after pip install
kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny  # 50x faster!
kin audio listen --engine whisper-cpp --model_size base               # Real-time
```

**Performance:** Up to 50x faster than OpenAI Whisper with low memory usage.
**✅ Automatic:** No separate installation required - included with `pywhispercpp` package.

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

View all available models:
```bash
kin audio models  # List all STT and TTS models
```

#### 🚀 API Server Models (Transformer-Based - Recommended)

All transformer-based TTS models must be used via API server:

```bash
# ✅ Fully Working Models
kin audio run kokoro-82m --port 8001      # 🎯 Best quality, natural voice, 320MB
kin audio run xtts-v2 --port 8002         # Voice cloning, multilingual, 1.8GB
kin audio run speecht5-tts --port 8004    # Microsoft SpeechT5, fast, 1.3GB
kin audio run bark-small --port 8005      # Suno Bark, expressive, 1.6GB

# ⚠️ Models Under Development (Not Yet Fully Implemented)
# kin audio run mms-tts-eng --port 8006     # Implementation in progress
# kin audio run tortoise-tts --port 8003    # Implementation in progress

# Use API
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output speech.wav
```

**Available TTS Models:**
- **kokoro-82m** - High-quality neural TTS (320MB) - 🔥 Recommended - ✅ **Fully Working**
- **xtts-v2** - Coqui XTTS v2, voice cloning (1.8GB) - ✅ **Fully Working**
- **speecht5-tts** - Microsoft SpeechT5 (1.3GB) - ✅ **Fully Working**
- **bark-small** - Suno Bark Small (1.6GB) - ✅ **Fully Working**
- **native** - System TTS (pyttsx3) - CLI only - ✅ **Works**
- **mms-tts-eng** - Meta MMS TTS English (1.2GB) - ⚠️ *Not yet implemented*
- **tortoise-tts** - High-quality multi-speaker (4.5GB) - ⚠️ *Not yet implemented*

#### 💻 CLI Direct Synthesis (Native Only)
```bash
# Only native/system TTS is supported for direct CLI synthesis
kin audio tts "Hello world" --model native        # ✅ Works - System TTS

# Transformer models are rejected with helpful guidance:
kin audio tts "Hello world" --model kokoro-82m    # ❌ Immediately shows error:
# "Error: Transformer-based TTS models must use the API server"
# "Please use: kin audio run kokoro-82m --port 8001"
# "Then call the API endpoint for synthesis"
```

**⚠️ Important:** CLI `kin audio tts` only works with `native` (system TTS). All transformer-based models (Kokoro, XTTS, SpeechT5, Bark, MMS, Tortoise) require the API server.

**✅ Correct Usage for Transformer Models:**
```bash
# Step 1: Start API server (one-time, stays loaded)
kin audio run kokoro-82m --port 8001

# Step 2: Use the API (instant responses)
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}' \
     --output speech.wav

# Or use Python API:
python -c "
from localkin_service_audio import synthesize_speech
result = synthesize_speech('Hello world')
"
```

**💡 Why API Server?**
- Model loads **once** and stays in memory (not on every request)
- Synthesis is **instant** after initial load
- Multiple requests reuse the loaded model
- Better resource management and error handling

### 🤖 LLM Integration

#### Voice-Based Conversational AI
```bash
# Requires Ollama running
kin audio listen --llm ollama --tts                    # Full voice AI
kin audio listen --llm ollama --llm-model qwen3:14b    # Custom model
kin audio listen --llm ollama --tts --stream           # Streaming responses
kin audio listen --engine whisper-cpp --model_size small --tts --tts-model kokoro-82m --vad --llm ollama --llm-model deepseek-r1:14b --stream                               # Custom models with Streaming
```

**Streaming Mode**: Add `--stream` for real-time LLM responses that speak as they generate, creating more natural conversational flow.

**Conversation Context**: LLM maintains conversation history during your session, allowing for contextual follow-up questions and natural dialogue flow.

## 📦 Installation & Setup

### 🎯 Prerequisites for Clean Machine Setup

Before installing LocalKin Service Audio, ensure you have these system tools:

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential build tools
xcode-select --install  # Install Xcode Command Line Tools

# Install dependencies
brew install openssl readline sqlite3 xz zlib tcl-tk ffmpeg
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt-get update

# Install essential build tools
sudo apt-get install -y build-essential curl git

# Install Python build dependencies
sudo apt-get install -y make libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev

# Install FFmpeg
sudo apt-get install -y ffmpeg
```

#### Windows (WSL2 Recommended)
```powershell
# Install WSL2
wsl --install

# Then follow Ubuntu/Debian instructions inside WSL2
```

---

### Step 1: Install Python 3.10+ with pyenv (Recommended)

#### macOS/Linux: Install pyenv
```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to your shell configuration (~/.zshrc or ~/.bashrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Reload your shell
source ~/.zshrc  # or source ~/.bashrc
```

**Note:** If you followed the Prerequisites section above, all required dependencies are already installed.

#### Install Python 3.10+ with LZMA Support
```bash
# Install Python 3.10 (or newer) with LZMA support
pyenv install 3.10.16

# Set as global Python version
pyenv global 3.10.16

# Verify LZMA support (critical for Kokoro TTS)
python -c "import lzma; print('✅ LZMA support OK')"

# Verify Python version
python --version
```

### Step 2: Install uv (Fast Package Manager)

```bash
# Install uv (highly recommended for better dependency management)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Step 3: Create Virtual Environment with uv

```bash
# Create a virtual environment for LocalKin
uv venv ~/.venv/localkin

# Activate the virtual environment
source ~/.venv/localkin/bin/activate  # On macOS/Linux
# Or on Windows: .venv\localkin\Scripts\activate

# Your prompt should now show (localkin)
```

### Step 4: Install LocalKin Service Audio

#### Option A: With uv (Recommended)
```bash
# Install LocalKin Service Audio with uv
uv pip install localkin-service-audio

# Verify installation
kin --version
```

#### Option B: With pip
```bash
# Install from PyPI
pip install localkin-service-audio

# Verify installation
kin --version
```

### Step 5: whisper.cpp (✅ Automatically Included!)

whisper.cpp is now automatically included via the `pywhispercpp` package - no manual installation needed!

```bash
# whisper.cpp models work immediately after pip install
kin audio transcribe audio.wav --engine whisper-cpp --model_size tiny

# Models download automatically on first use - no setup required!
```

**Note:** If you need the original C++ executable for advanced use cases, you can still install it manually as before.

### Step 6: Verify FFmpeg Installation

```bash
# FFmpeg should already be installed from Prerequisites
# Verify it's working
ffmpeg -version
```

**Note:** If you skipped the Prerequisites section, install FFmpeg now:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`

### Step 7: Install Ollama (Optional - for Voice AI Features)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model (in another terminal)
ollama pull qwen2.5:3b  # or any model you prefer

# Verify
ollama list
```

### Quick Start After Setup

```bash
# Activate your virtual environment
source ~/.venv/localkin/bin/activate

# Check available models
kin audio models

# Test with included sample audio
python -c "import localkin_service_audio as lsa; print(f'Sample audio: {lsa.get_sample_audio_path()}')"

# Try speech recognition with the sample
python3 << EOF
import localkin_service_audio as lsa
result = lsa.transcribe_audio(lsa.get_sample_audio_path())
print(f"Transcription: {result}")
EOF

# Or use CLI with your own audio file
kin audio transcribe audio.wav --engine whisper-cpp

# Start Kokoro TTS server
kin audio run kokoro-82m --port 8001

# Voice AI conversation (requires Ollama)
kin audio listen --llm ollama --tts --stream
```

### Alternative: Quick Install (Without Virtual Environment)

If you prefer a simpler setup without virtual environments:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install LocalKin directly
uv pip install localkin-service-audio

# Verify installation
kin --version
```

**Why uv?** It ensures proper Python environment with LZMA support needed for Kokoro TTS, avoiding common pyenv-related issues.

### Install from Source (For Contributors/Advanced Users)
```bash
# Clone repository for development or advanced setup
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio

# Install with uv
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

#### Kokoro TTS "Could not import module 'pipeline'" Error
If you get this error when trying to use Kokoro TTS:
```
ERROR: Could not import module 'pipeline'. Are this object's requirements defined correctly?
```

This means your Python installation is missing LZMA compression support. Here are the solutions:

**Option 1: Use System Python (Recommended - Quickest)**
```bash
# System Python has LZMA support built-in
/usr/bin/python3 -m pip install --user localkin-service-audio

# Add to your PATH or create an alias
echo 'alias kin="/usr/bin/python3 -m localkin_service_audio.cli"' >> ~/.zshrc
source ~/.zshrc

# Now run normally
kin audio run kokoro-82m --port 8001
```

**Option 2: Use uv with System Python**
```bash
# Use uv with system Python (keeps LZMA support)
/usr/bin/python3 -m pip install uv

# Create a virtual environment with system Python
/usr/bin/python3 -m venv ~/.venv/localkin
source ~/.venv/localkin/bin/activate

# Install with uv
uv pip install localkin-service-audio

# Run kokoro TTS
kin audio run kokoro-82m --port 8001
```

**Option 3: Use SpeechT5 Instead (No Python changes needed)**
```bash
# SpeechT5 works without LZMA - use with your current Python
kin audio run speecht5-tts --port 8001
```

**Option 4: Fix pyenv Python (Advanced - requires reinstalling all packages)**
```bash
# Only if you really want to fix pyenv Python
# WARNING: This removes all installed packages!

# Install LZMA library first
brew install xz

# List your packages to reinstall later
pip freeze > requirements_backup.txt

# Reinstall Python with LZMA support
pyenv uninstall 3.10.0
pyenv install 3.10.0

# Reinstall packages
pip install -r requirements_backup.txt
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
