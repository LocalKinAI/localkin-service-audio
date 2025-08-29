# OllamaAudio 🎵

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Local Speech-to-Text and Text-to-Speech with OllamaAudio

**OllamaAudio** simplifies local deployment of **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** models. An intuitive **local audio** tool inspired by **Ollama's** simplicity - perfect for **local audio processing** workflows with CLI support.

---

## ✨ Features

- **🚀 Fast Startup**: Instant application launch with lazy loading architecture
- **🎯 Multiple STT Engines**: OpenAI Whisper, Ollama-based whisper, and Hugging Face models
- **🔊 Multiple TTS Engines**: Native OS TTS, Ollama-based conversational models, and SpeechT5/Bark
- **🌐 REST API Server**: Run models as API servers with automatic endpoints
- **📦 Smart Model Management**: Auto-pull models when needed, intelligent caching
- **💾 Persistent Cache**: Local model storage with size tracking and cleanup
- **🔄 Auto-Pull**: Models automatically download when running if not cached
- **📊 Real-Time Status**: Live model status tracking with emoji indicators
- **⚡ Performance Optimized**: Memory-efficient with GPU acceleration support
- **🎨 Professional Results**: High-quality audio processing with fine-tuned control
- **🌐 CLI-First**: Simple command-line interface inspired by Ollama

## 🚀 Quick Start

### Option 1: Install with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio

# Install with uv (fast and reliable)
uv pip install -e .

# Now you can use ollamaaudio directly
ollamaaudio --help
```

### Option 2: Install with uv venv (Isolated Environment)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Linux/macOS
# On Windows: .venv\Scripts\activate

# Install ollamaaudio
uv pip install -e .

# Use ollamaaudio
ollamaaudio --help
```

### Option 3: Traditional pip Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio

# Install dependencies (slower than uv)
pip install -r ollamaaudio/requirements.txt
pip install -e .

# Run commands using the module
python -m ollamaaudio.cli --help
```

### Basic Usage

```bash
# Check version and help
ollamaaudio --version
ollamaaudio --help

# List all available models with status
ollamaaudio list

# Speech-to-Text with local Whisper
ollamaaudio stt audio.wav

# Text-to-Speech with native engine
ollamaaudio tts "Hello, world!"

# 🚀 NEW: Run models as API servers (auto-pulls if needed)
ollamaaudio run whisper-tiny-hf --port 8000
ollamaaudio run speecht5-tts --port 8001

# Check system status and cache
ollamaaudio status
ollamaaudio cache info

# Manage model cache
ollamaaudio cache clear whisper-tiny-hf  # Clear specific model
ollamaaudio cache clear                   # Clear all cached models
```

### 🚀 API Server Quick Start

```bash
# Start a Whisper STT server (auto-pulls if needed)
ollamaaudio run whisper-tiny-hf --port 8000

# Server starts at http://localhost:8000
# API docs: http://localhost:8000/docs

# Use the API (in another terminal)
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"

# Response:
# {
#   "text": "Hello, this is a test transcription",
#   "language": "en",
#   "confidence": 0.95
# }
```

---

## 🎯 Supported Models

Choose from a variety of state-of-the-art audio processing models:

### 📊 Model Status Overview

```bash
$ ollamaaudio list
MODEL                     TYPE   STATUS             SOURCE          DESCRIPTION
------------------------------------------------------------------------------------------
whisper                   stt    📦 Local Library    openai-whisper  Local transcription
whisper-tiny-hf           stt    ✅ Pulled           huggingface     OpenAI Whisper Tiny
whisper-base-hf           stt    ✅ Pulled           huggingface     OpenAI Whisper Base
whisper-large-v2-hf       stt    ⬇️ Not Pulled      huggingface     OpenAI Whisper Large v2
speecht5-tts              tts    ⬇️ Not Pulled      huggingface     Microsoft SpeechT5 TTS
bark-small                tts    ⬇️ Not Pulled      huggingface     Suno Bark small TTS
native                    tts    📦 Local Library    pyttsx3         Uses macOS native TTS
whisper-large-v3          stt    ⬇️ Not Pulled      ollama          Advanced Whisper Large v3
whisper-base              stt    ⬇️ Not Pulled      ollama          Whisper Base model
llama3.2:3b-instruct-q4_0 tts    ⬇️ Not Pulled      ollama          Llama 3.2 3B model
qwen2.5:3b-instruct-q4_0  tts    ⬇️ Not Pulled      ollama          Qwen 2.5 3B model
mistral:7b-instruct-q4_0  tts    ⬇️ Not Pulled      ollama          Mistral 7B model
```

### 📋 Model Specifications

| Model | Type | Source | License | Quality | Speed | Size | Status |
|-------|------|--------|---------|---------|-------|------|---------|
| **whisper** | STT | Local | MIT | High | Fast | 139MB-1550MB | ✅ Ready |
| **whisper-tiny-hf** | STT | Hugging Face | MIT | Basic | 32x | 39MB | ✅ Cached |
| **whisper-base-hf** | STT | Hugging Face | MIT | Good | 16x | 290MB | ✅ Cached |
| **whisper-large-v2-hf** | STT | Hugging Face | MIT | Excellent | 1x | 2.87GB | ⬇️ Not Pulled |
| **speecht5-tts** | TTS | Hugging Face | MIT | High | Medium | 250MB | ⬇️ Not Pulled |
| **bark-small** | TTS | Hugging Face | MIT | Very High | Slow | 1.7GB | ⬇️ Not Pulled |
| **native** | TTS | Local | MIT | Good | Instant | Local | ✅ Ready |
| **whisper-large-v3** | STT | Ollama | Apache 2.0 | Very High | Medium | 1550MB | ⬇️ Not Pulled |
| **whisper-base** | STT | Ollama | Apache 2.0 | Good | Very Fast | 139MB | ⬇️ Not Pulled |
| **llama3.2:3b** | TTS | Ollama | Llama License | High | Medium | 1820MB | ⬇️ Not Pulled |
| **qwen2.5:3b** | TTS | Ollama | Apache 2.0 | High | Medium | 1820MB | ⬇️ Not Pulled |
| **mistral:7b** | TTS | Ollama | Apache 2.0 | Very High | Slow | 3820MB | ⬇️ Not Pulled |

### STT Models - Speech-to-Text

#### 🚀 Hugging Face Models (API Server - Recommended)
```bash
# Start API server (auto-pulls if needed)
ollamaaudio run whisper-tiny-hf --port 8000    # Fastest, 39MB
ollamaaudio run whisper-base-hf --port 8001    # Balanced, 290MB
ollamaaudio run whisper-large-v2-hf --port 8002 # Best quality, 2.87GB

# Use the API
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"

# Interactive API docs: http://localhost:8000/docs
```

#### Local Whisper Models (Direct CLI - No API Key Required)
```bash
# Use different Whisper model sizes
ollamaaudio stt audio.wav --model_size tiny    # Fastest, ~39 MB
ollamaaudio stt audio.wav --model_size base    # Default, ~139 MB
ollamaaudio stt audio.wav --model_size small   # Better quality, ~461 MB
ollamaaudio stt audio.wav --model_size medium  # High quality, ~1.42 GB
ollamaaudio stt audio.wav --model_size large   # Best quality, ~2.87 GB
```

#### Ollama Whisper Models (Requires Ollama)
```bash
# Pull models first
ollamaaudio pull whisper-large-v3
ollamaaudio pull whisper-base

# Use them (future feature)
ollamaaudio stt audio.wav --model whisper-large-v3
```

### TTS Models - Text-to-Speech

#### 🚀 Hugging Face Models (API Server - Recommended)
```bash
# Start TTS API server (auto-pulls if needed)
ollamaaudio run speecht5-tts --port 8001    # Microsoft SpeechT5, 250MB
ollamaaudio run bark-small --port 8002      # Suno Bark, 1.7GB

# Use the API for speech synthesis
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is generated speech!",
       "speaker": "male"
     }' \
     --output speech.wav

# Interactive API docs: http://localhost:8001/docs
```

#### Native OS TTS (Works Immediately)
```bash
# Basic usage
ollamaaudio tts "Hello, world!"

# Save to file
ollamaaudio tts "This is a test" --output hello.wav
```

#### Ollama Conversational Models (Future Feature)
```bash
# Pull conversational models for TTS
ollamaaudio pull llama3.2:3b-instruct-q4_0
ollamaaudio pull qwen2.5:3b-instruct-q4_0

# Use them for conversational TTS (future feature)
ollamaaudio tts "Tell me a story" --model llama3.2:3b-instruct-q4_0
```

---

## 📦 Installation & Setup

### Prerequisites

- **Python 3.8+**
- **uv** (fast Python package installer)
- **Ollama** (optional, for Ollama-based models)
- **FFmpeg** (for audio processing)

### Quick Install with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio

# Install with uv (fast and reliable)
uv pip install -e .
```

### Ollama Setup (Optional)

For advanced models, install Ollama:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### Audio Processing Setup

For audio file processing, ensure you have audio libraries (all included in pyproject.toml):

```bash
# Install additional audio processing dependencies if needed
uv pip install librosa soundfile

# For macOS, you might need additional audio support
# All necessary dependencies are included in pyproject.toml
```

---

## 🎛️ Command Reference

### Core Commands

```bash
# Display help
ollamaaudio --help

# Show version
ollamaaudio --version

# List all available models with status
ollamaaudio list

# Check system status
ollamaaudio status
```

### 🚀 Model Server Commands

```bash
# Run model as API server (auto-pulls if needed)
ollamaaudio run whisper-tiny-hf --port 8000     # STT server
ollamaaudio run speecht5-tts --port 8001       # TTS server

# Server endpoints:
# - http://localhost:8000/           # API info
# - http://localhost:8000/health     # Health check
# - http://localhost:8000/docs       # Interactive API docs
# - http://localhost:8000/transcribe # STT endpoint
# - http://localhost:8000/synthesize # TTS endpoint
```

### Model Management

```bash
# Pull models (Ollama or Hugging Face)
ollamaaudio pull whisper-large-v3              # Ollama model
ollamaaudio pull whisper-tiny-hf               # Hugging Face model

# Cache management
ollamaaudio cache info                         # Show cache status
ollamaaudio cache clear whisper-tiny-hf       # Clear specific model
ollamaaudio cache clear                       # Clear all cached models
```

### Speech-to-Text

```bash
# Basic transcription
ollamaaudio stt audio.wav

# Specify model size for Whisper
ollamaaudio stt audio.wav --model_size large

# Use specific model (future feature)
ollamaaudio stt audio.wav --model whisper-large-v3

# Save transcription to file
ollamaaudio stt audio.wav > transcription.txt
```

### Text-to-Speech

```bash
# Basic speech synthesis
ollamaaudio tts "Hello, world!"

# Save to audio file
ollamaaudio tts "This is a test message" --output test.wav

# Use specific model (future feature)
ollamaaudio tts "Hello" --model llama3.2:3b-instruct-q4_0

# Read from file
ollamaaudio tts "$(cat script.txt)" --output narration.wav
```

---

## 💾 Cache Management

OllamaAudio uses intelligent caching to store downloaded models locally for faster subsequent loads.

### Cache Location

Models are cached in: `~/.ollamaaudio/cache/huggingface/`

### Cache Commands

```bash
# View cache status
ollamaaudio cache info

# Output:
# 📦 Cached Models (2):
#   • whisper-tiny-hf (580.65MB)
#   • whisper-base-hf (1112.27MB)

# Clear specific model from cache
ollamaaudio cache clear whisper-tiny-hf

# Clear all cached models
ollamaaudio cache clear

# Check cache directory size
du -sh ~/.ollamaaudio/cache/
```

### Auto-Pull Behavior

When you run a model that isn't cached:

```bash
ollamaaudio run whisper-large-v2-hf --port 8000
# Output:
# 📥 Model 'whisper-large-v2-hf' not found in cache. Pulling it first...
# 📥 Downloading model from Hugging Face: openai/whisper-large-v2
# ✅ Model downloaded to: /Users/.../whisper-large-v2-hf
# ✅ Model pulled successfully!
# 🚀 Starting OllamaAudio API server for whisper-large-v2-hf
```

### Cache Benefits

- **🚀 Fast Startup**: Pre-downloaded models load instantly
- **💾 Space Efficient**: Only cache models you actually use
- **🔄 Auto Management**: Automatic downloading when needed
- **📊 Status Tracking**: Real-time cache status in `list` command

---

## ⚙️ Configuration

### Model Configuration

Models are configured in `ollamaaudio/models.json`. The system supports:

- **Local models**: OpenAI Whisper, pyttsx3 (native TTS)
- **Hugging Face models**: Transformers-based STT/TTS models
- **Ollama models**: Whisper variants, conversational models
- **Custom models**: Easy to add new model configurations

### Configuration File Structure

```json
{
  "models": [
    {
      "name": "whisper-tiny-hf",
      "type": "stt",
      "description": "OpenAI Whisper Tiny model from Hugging Face",
      "source": "huggingface",
      "huggingface_repo": "openai/whisper-tiny",
      "license": "MIT",
      "requirements": ["transformers", "torch"],
      "sizes": ["39MB"],
      "tags": ["fast", "lightweight"]
    },
    {
      "name": "speecht5-tts",
      "type": "tts",
      "description": "Microsoft SpeechT5 TTS model from Hugging Face",
      "source": "huggingface",
      "huggingface_repo": "microsoft/speecht5_tts",
      "license": "MIT",
      "requirements": ["transformers", "torch"],
      "sizes": ["250MB"],
      "tags": ["high-quality", "multi-speaker"]
    }
  ],
  "metadata": {
    "version": "1.0.0",
    "last_updated": "2024-12-19"
  }
}
```

### Adding Custom Models

1. **Edit `ollamaaudio/models.json`**
2. **Add your model configuration** with Hugging Face repo details
3. **Test with `ollamaaudio list`** to see status
4. **Run with `ollamaaudio run your-model --port 8000`**

### Model Sources

| Source | Description | Auto-Pull | API Server |
|--------|-------------|-----------|------------|
| `huggingface` | Hugging Face Hub models | ✅ Yes | ✅ Yes |
| `ollama` | Ollama models | ✅ Yes | 🚧 Future |
| `openai-whisper` | Local Whisper models | ❌ N/A | ❌ No |
| `pyttsx3` | Native OS TTS | ❌ N/A | ❌ No |

---

## 🔧 Advanced Usage

### Batch Processing

```bash
# Process multiple audio files
for file in *.wav; do
  ollamaaudio stt "$file" > "${file%.wav}.txt"
done

# Batch TTS generation
echo "First message" > messages.txt
echo "Second message" >> messages.txt
echo "Third message" >> messages.txt

while IFS= read -r message; do
  ollamaaudio tts "$message" --output "tts_$(echo "$message" | head -c 10).wav"
done < messages.txt
```

### Audio Format Support

OllamaAudio supports various audio formats through Whisper:

- **WAV**: Native support
- **MP3**: Requires FFmpeg
- **M4A**: Requires FFmpeg
- **FLAC**: Native support
- **OGG**: Requires FFmpeg

### Quality vs Speed Trade-offs

| Model Size | Quality | Speed | Use Case |
|------------|---------|-------|----------|
| tiny | Basic | 32x | Quick transcription |
| base | Good | 16x | Balanced default |
| small | High | 8x | Quality important |
| medium | Very High | 4x | Professional work |
| large | Excellent | 1x | Best possible quality |

---

## 🚀 Performance & Hardware

### Minimum Requirements

- **RAM**: 4GB system RAM
- **Storage**: 500MB free space (for base models)
- **Python**: 3.8+

### Recommended Hardware

#### For Basic Usage (Whisper Tiny/Base)
- **RAM**: 8GB+ system RAM
- **Storage**: 2GB+ free space
- **GPU**: Optional (CPU works fine)

#### For High-Quality Models (Whisper Medium/Large)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA/AMD GPU with 4GB+ VRAM (recommended)

#### For Ollama Models
- **RAM**: 8GB+ system RAM
- **Storage**: 5GB+ free space per model
- **GPU**: 8GB+ VRAM for larger models

### Platform Support

- ✅ **macOS**: Full support (Intel/Apple Silicon)
- ✅ **Linux**: Full support
- ✅ **Windows**: Full support
- ✅ **CPU-only**: All platforms
- ✅ **GPU acceleration**: CUDA (NVIDIA), MPS (Apple), ROCm (AMD)

---

## 🔧 Troubleshooting

### Common Issues

#### "Model not found" Error
```bash
# Check available models
ollamaaudio list

# For Ollama models, ensure Ollama is running
ollama serve

# Pull the model first
ollamaaudio pull whisper-base
```

#### Audio File Issues
```bash
# Check if file exists and is readable
ls -la audio.wav

# Try converting audio format
ffmpeg -i input.mp3 output.wav
ollamaaudio stt output.wav
```

#### Memory Issues
```bash
# Use smaller models
ollamaaudio stt audio.wav --model_size tiny

# Close other applications
# Use CPU-only mode if GPU memory is limited
```

#### Permission Issues
```bash
# Ensure proper permissions
chmod +x ollamaaudio
# or
python -m ollamaaudio.cli
```

### Debug Mode

```bash
# Enable verbose output
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
ollamaaudio stt audio.wav
```

---

## 🌐 REST API Reference

OllamaAudio provides complete REST APIs for both STT and TTS models when running in server mode.

### 🚀 Getting Started with API

```bash
# Start an API server
ollamaaudio run whisper-tiny-hf --port 8000

# Interactive API documentation
open http://localhost:8000/docs

# API information
curl http://localhost:8000/
```

### 📋 API Endpoints

#### Common Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API server information and available endpoints |
| `GET` | `/health` | Health check and server status |
| `GET` | `/models` | Information about loaded models |
| `GET` | `/docs` | **Interactive API documentation (Swagger UI)** |

#### STT Endpoints (Whisper Models)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcribe` | Upload audio file for transcription |

**Transcription Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav" \
     -F "language=en" \
     -F "task=transcribe"
```

**Response:**
```json
{
  "text": "Hello, this is the transcription result.",
  "language": "en",
  "confidence": 0.95
}
```

#### TTS Endpoints (SpeechT5/Bark Models)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/synthesize` | Generate speech from text |

**Synthesis Request:**
```bash
curl -X POST "http://localhost:8001/synthesize" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, this is generated speech!",
       "speaker": "male"
     }' \
     --output speech.wav
```

### 📊 Multiple Servers

Run multiple models on different ports:

```bash
# Terminal 1: STT Server
ollamaaudio run whisper-tiny-hf --port 8000

# Terminal 2: TTS Server
ollamaaudio run speecht5-tts --port 8001

# Terminal 3: Use both APIs
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@input.wav" > transcription.json

curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello from OllamaAudio!"}' \
     --output output.wav
```

### 🔧 Python API (Future Feature)

```python
from ollamaaudio.core import OllamaAudio

# Initialize
audio = OllamaAudio()

# Speech-to-Text
transcription = audio.transcribe("audio.wav", model="whisper-base")

# Text-to-Speech
audio.synthesize("Hello world", output_file="hello.wav")

# List models
models = audio.list_models()
```

### ⚙️ Configuration API

```python
from ollamaaudio.config import get_models, find_model

# Get all models
models = get_models()

# Find specific model
model = find_model("whisper")

# Get model sizes
sizes = get_model_sizes("whisper")
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install with uv
uv venv
source .venv/bin/activate  # Linux/macOS
# or on Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting (uv handles the tools automatically)
ruff check ollamaaudio/
ruff format ollamaaudio/
```

### Adding New Models

1. **Update `models.json`**: Add model configuration
2. **Implement handler**: Add model-specific logic in appropriate module
3. **Update CLI**: Add command-line options if needed
4. **Add tests**: Create unit tests for the new functionality
5. **Update docs**: Add documentation and examples

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI**: For Whisper models and the transformers library
- **Ollama**: For the inspiration and Ollama models
- **Hugging Face**: For the model hub and transformers ecosystem
- **Microsoft**: For SpeechT5 and other TTS models
- **Suno**: For Bark TTS models
- **PyTorch**: For the deep learning framework
- **FastAPI**: For the REST API framework
- **uv**: For fast Python package management
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ollamaaudio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ollamaaudio/discussions)

---

## 🚀 Quick Examples

### Start Your First API Server

```bash
# Install with uv
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/yourusername/ollamaaudio.git
cd ollamaaudio
uv venv && source .venv/bin/activate
uv pip install -e .

# Start a Whisper STT server (auto-pulls if needed)
ollamaaudio run whisper-tiny-hf --port 8000

# In another terminal, transcribe audio
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@your-audio.wav"

# Interactive API docs
open http://localhost:8000/docs
```

### Multiple Models, Multiple Servers

```bash
# Terminal 1: STT Server
ollamaaudio run whisper-base-hf --port 8000

# Terminal 2: TTS Server
ollamaaudio run speecht5-tts --port 8001

# Terminal 3: Use both
# Transcribe audio to text
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@input.wav" \
     | jq -r '.text' > text.txt

# Generate speech from text
curl -X POST "http://localhost:8001/synthesize" \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"$(cat text.txt)\"}" \
     --output output.wav
```

### Check Status and Cache

```bash
# See all models with status
ollamaaudio list

# Check cache usage
ollamaaudio cache info

# System status
ollamaaudio status
```

---

## 🎯 What's New in v1.0

- ✅ **REST API Servers**: Run models as complete API services
- ✅ **Auto-Pull**: Models download automatically when needed
- ✅ **Smart Caching**: Persistent local storage with status tracking
- ✅ **Hugging Face Integration**: Direct integration with HF Hub
- ✅ **Multiple Model Support**: STT + TTS in parallel servers
- ✅ **Interactive Documentation**: Auto-generated API docs
- ✅ **Status Indicators**: Real-time model status with emojis
- ✅ **Fast Package Management**: uv integration for quick installs

---

**🎉 Ready to get started with local audio AI?** Install OllamaAudio and run your first API server in minutes! 🎵✨

```bash
ollamaaudio run whisper-tiny-hf --port 8000
# 🚀 API server ready at http://localhost:8000
```
