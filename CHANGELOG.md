# Changelog

All notable changes to LocalKin Service Audio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.8] - 2026-02-15

### Fixed
- **Chinese TTS missing dependency**: Added `ordered-set` to dependencies — required by misaki (kokoro's text processor) for Chinese voice pipelines but not declared in its own deps

---

## [2.0.7] - 2026-02-14

### Fixed
- **Spacy model download in uv environments**: `en-core-web-sm` is not on PyPI — previous install attempts all failed silently. Now downloads the wheel directly from spacy's GitHub releases using the correct URL for the installed spacy version.

---

## [2.0.6] - 2026-02-14

### Fixed
- **Startup crash on fresh install**: `kin -V` and all commands failed with import errors due to `musicgen_strategy.py` eagerly importing `torch` and `transformers` at module level
  - Moved to lazy imports inside `load()` method (matching heartmula_strategy.py pattern)
  - Guarded music imports in `__init__.py` with try/except
- **Kokoro TTS in uv environments**: Fixed `No module named pip` error when downloading spacy model
  - Now tries `uv pip install` first, falls back to `pip`, then `spacy download`
- Added troubleshooting for `numpy.dtype size changed` binary incompatibility (numpy/pandas/scikit-learn mismatch)

---

## [2.0.4] - 2026-02-14

### Added
- **Music Generation**: New `kin audio music` command group with two engines
  - **MusicGen** (Meta): Text-to-music with small/medium/large model sizes (5–30s)
  - **HeartMuLa** (Open Source): Multilingual music generation with Chinese lyrics support, tag-based style control, and durations up to 240s
- `kin audio music generate <prompt>` — generate music from text descriptions
  - `--model musicgen:small|medium|large` or `--model heartmula:3b|7b`
  - `--tags "piano,romantic,wedding"` — style tags (HeartMuLa)
  - `--duration`, `--temperature`, `--device`, `--output`, `--play/--no-play`
- `kin audio music models` — list available music models with requirements
- **HeartMuLa auto-install**: `heartlib` is automatically installed from GitHub on first use (via `kin audio music generate --model heartmula:*` or `kin audio pull heartmula:3b`)
- **HeartMuLa in model registry**: `heartmula:3b` and `heartmula:7b` now registered, enabling `kin audio pull heartmula:3b`
- `HeartMuLaStrategy` exported from package (`from localkin_service_audio import HeartMuLaStrategy`)
- Music generation unit tests in `tests/unit/test_music.py`

### Fixed
- **HeartMuLa MPS support**: Fixed `invalid type: 'torch.mps.FloatTensor'` crash when generating music on Apple Silicon
  - HeartCodec now runs on CPU during detokenize (unified memory — no copy overhead)
  - Patched heartlib's `_unload()` which had hardcoded `torch.cuda` calls that crash on MPS
- **Dependency upgrades**: Bumped `torch>=2.6.0`, `torchaudio>=2.6.0`, removed `transformers<4.50` upper pin
  - Fixes `torch.load` security check (CVE-2025-32434) that blocked MusicGen medium/large loading
  - MusicGen small was unaffected (has safetensors), medium/large only had `.bin` weights
- `MusicGenStrategy.unload()` referenced non-existent `self.pipe` attribute (now uses `self.model`)
- `test_music.py` integration test used wrong `ModelConfig` parameter names
- HeartMuLa checkpoint downloads now use `$LOCALKIN_HOME/cache/heartmula/` instead of `~/.cache/heartmula`
- Updated stale Kokoro error message that still referenced `transformers<4.50`

### Changed
- README now recommends `uv` as primary installer (10-100x faster for heavy ML deps)
- Added upgrade instructions for existing users (torch 2.4 → 2.6)

---

## [2.0.1] - 2026-02-05

### Added
- **Restored CLI commands from v1.x**: `status`, `cache`, `ps`, `add-model`, `list-templates`
  - `kin audio status` - System health check (library availability, model registry, cache)
  - `kin audio cache info` - Show cached models and sizes
  - `kin audio cache clear [model]` - Clear model cache with confirmation
  - `kin audio ps` - Show running LocalKin Audio servers on ports 8000-8005
  - `kin audio add-model` - Register models from templates or HuggingFace repos
  - `kin audio list-templates` - List available model templates
- **`-h` and `-V` short flags** for help and version
- **`kin audio config set`** subcommand for changing settings from CLI
- **`LOCALKIN_HOME` environment variable** to relocate all data (cache, config, models) to another disk
- **Kokoro multilingual TTS**: 54 voices across 9 languages (en, es, fr, hi, it, ja, pt, zh)
  - Auto-selects correct Kokoro language pipeline based on voice prefix
  - Chinese voices: `zf_xiaoxiao`, `zf_xiaobei`, `zm_yunyang`, etc.
  - Japanese voices: `jf_alpha`, `jf_nezumi`, `jm_kumo`, etc.
  - French, Spanish, Italian, Hindi, Portuguese voices
- **Moonshine STT** added to model registry (`moonshine:tiny`, `moonshine:base`) — ultra-fast English ASR
- **Model availability status** in `kin audio models`: shows ✅ Ready / 📦 Not installed / 🔮 Planned
- **28 total models** registered (20 STT + 8 TTS) with clear status indicators
- **`kin web`** web interface working (fixed FastAPI app creation and Jinja2 template errors)
- Unit tests for all 5 restored commands (114 total tests)

### Fixed
- **`kin web` import error**: was importing non-existent `app` from `ui.routes`; now builds FastAPI app from `create_ui_router()`
- **`kin web` template error**: Jinja2 `||` syntax and undefined `stats` variable in `modern_index.html`
- **Kokoro TTS 404 error**: Default voice `af` no longer exists on HuggingFace; changed to `af_heart`
- All hardcoded `~/.localkin-service-audio` paths now respect `LOCALKIN_HOME` (`core/models.py`, `core/config/model_registry.py`, `api/server.py`)
- `kin audio config` now uses Settings singleton (was showing wrong cache path)
- Relaxed `numpy` version constraint (removed `<1.25.0` upper bound) for compatibility with modern PyTorch/scipy
- Moved `TTS` (Coqui) from core dependencies to optional `[tts]` extra to fix pip install failures

---

## [2.0.0] - 2026-02-04

### 🎉 Major Release: Voice AI Platform with Chinese Language Support

Complete architectural overhaul transforming LocalKin into a comprehensive Voice AI Platform
with Strategy Pattern architecture, Chinese language support, and Claude MCP integration.

### ✨ New Architecture

- **AudioEngine facade** - Unified interface for all audio operations (inspired by ollamadiffuser)
- **Strategy Pattern** - Pluggable STT/TTS engines via `STTStrategy` and `TTSStrategy` base classes
- **ModelRegistry** - Centralized model configuration with external YAML/JSON support
- **Rich data types** - `TranscriptionResult`, `AudioResult`, `VoiceInfo`, `ModelConfig`

### 🎤 New STT Engines

| Engine | Languages | Speed | Key Feature |
|--------|-----------|-------|-------------|
| **SenseVoice** | 50+ (zh/en/ja/ko) | 15x Whisper | Emotion detection |
| **Paraformer** | Chinese | Very Fast | VAD + Punctuation |
| **Moonshine** | English | 5x real-time | Ultra-lightweight |
| Parakeet TDT | English | >2000x RT | NVIDIA fastest |
| Canary | Multilingual | Fast | #1 HF leaderboard |

### 🔊 New TTS Engines

| Engine | Languages | Key Feature |
|--------|-----------|-------------|
| **CosyVoice 2** | zh/en/ja/ko/yue | SOTA Chinese, voice cloning |
| **ChatTTS** | zh/en | Conversational, emotion tokens |
| **F5-TTS** | en/zh | Zero-shot voice cloning |

### 🖥️ New CLI (Click-based)

```bash
kin audio transcribe audio.wav --model whisper-cpp:base
kin audio tts "Hello world" --model kokoro
kin audio listen             # Real-time voice with TTS/LLM
kin audio listen --llm ollama --tts --stream  # Voice AI conversation
kin audio models --language zh
kin audio recommend          # Hardware-aware recommendations
kin audio config             # View/manage configuration
kin audio benchmark audio.wav -m whisper-cpp:base -m sensevoice:small
kin mcp                      # Start MCP server for Claude
```

### 🤖 MCP Server (Claude Integration)

Tools available when using `kin mcp`:
- `transcribe_audio` - Transcribe audio files
- `synthesize_speech` - Generate speech from text
- `clone_voice` - Clone voice from reference audio
- `list_models` / `list_voices` - Query available models

### 📡 WebSocket Streaming API

- `/api/stream` - Real-time streaming transcription
- `/api/stream/tts` - Streaming text-to-speech
- Voice activity detection with partial results

### 🧪 Test Suite

- Unit tests for core types and dataclasses
- Unit tests for AudioEngine and strategies
- Unit tests for ModelRegistry
- Unit tests for CLI commands (28 tests)
- 99 total unit tests with pytest

### 📦 New Optional Dependencies

```bash
pip install localkin-service-audio[chinese]     # SenseVoice, Paraformer, CosyVoice
pip install localkin-service-audio[cloning]     # F5-TTS voice cloning
pip install localkin-service-audio[fast]        # Moonshine ultra-fast
pip install localkin-service-audio[mcp]         # Claude MCP integration
pip install localkin-service-audio[all-models]  # Everything
```

### 🔄 Migration from v1.x

**CLI:**
```bash
# Old: kin audio transcribe audio.wav --engine whisper-cpp --model_size base
# New: kin audio transcribe audio.wav --model whisper-cpp:base
```

**Python API:**
```python
# Old
from localkin_service_audio.core import transcribe_audio
text = transcribe_audio("base", "audio.wav", engine="whisper-cpp")

# New
from localkin_service_audio.core import get_audio_engine
engine = get_audio_engine()
engine.load_stt("whisper-cpp:base")
result = engine.transcribe("audio.wav")
print(result.text, result.language, result.emotion)
```

---

## [1.1.3] - 2024-10-02

### Fixed
- Fixed whisper-cpp integration issues
- Improved pywhispercpp compatibility

---

## [1.1.0] - 2024-09-20

### Added
- whisper.cpp support via pywhispercpp (50x faster)
- Voice Activity Detection (VAD) support
- Batched inference for long audio files

---

## [1.0.0] - 2024-08-29

### Added
- Initial release
- OpenAI Whisper, faster-whisper integration
- pyttsx3, Kokoro, XTTS TTS engines
- FastAPI server and Web UI
- CLI with argparse
