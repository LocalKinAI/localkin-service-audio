# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-XX

### Added
- **Streaming LLM Integration**: Real-time conversation with `--stream` flag
- **Conversation Context**: LLM maintains chat history during sessions
- **whisper.cpp Support**: Ultra-fast C++ Whisper implementation
- **Multiple STT Engines**: OpenAI Whisper, faster-whisper, whisper.cpp
- **Multiple TTS Engines**: Native OS TTS, SpeechT5, Bark, Kokoro, XTTS
- **REST API Server**: Run models as API servers with automatic endpoints
- **Modern Web Interface**: Beautiful web UI with file upload support
- **Voice Activity Detection (VAD)**: Enhanced silence filtering
- **Model Management**: Auto-pull models, intelligent caching
- **Cross-platform Support**: Works on macOS, Linux, Windows

### Features
- Real-time STT/TTS loop with `kin audio listen`
- File transcription with `kin audio transcribe`
- Text-to-speech synthesis with `kin audio tts`
- Model management with `kin audio models`
- API server mode for production use
- Web interface for easy access

### Dependencies
- Python 3.10+
- OpenAI Whisper, faster-whisper, whisper.cpp
- PyTorch, Transformers, Hugging Face Hub
- FastAPI, Uvicorn for API server
- SoundDevice, SciPy for audio processing
- Pyttsx3 for native TTS
