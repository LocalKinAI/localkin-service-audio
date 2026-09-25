# LocalKin Service Audio

[![PyPI version](https://badge.fury.io/py/localkin-service-audio.svg)](https://pypi.org/project/localkin-service-audio/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Local Voice AI Platform** - Speech-to-Text and Text-to-Speech with Chinese language support, voice cloning, and Claude integration via MCP.

## What's New in v2.1.1

- **Songs with lyrics**: `kin audio music generate --model minimax-music3 --lyrics @song.txt` — MiniMax Music 3 on MLX (29 s of song in about 3 minutes on an M3 Ultra).
- **Music from ComfyUI**: ACE-Step 1.5, YuE2, Stable Audio 3 — or any audio blueprint ComfyUI has — run through a local or remote ComfyUI with the weights already installed there.
- `kin audio music models` shows the same table as `kin audio models`, with whether each model's weights are installed.

## What's New in v2.1.0

- **50 current models, one name each, on any machine.** Qwen3-ASR, Fun-ASR-Nano, FireRedASR2, Parakeet, Nemotron, VibeVoice-ASR; Qwen3-TTS, VoxCPM2, CosyVoice3, IndexTTS-2, Fish Audio S2 Pro, Chatterbox and more — picked by Hugging Face downloads, likes and trending. On a Mac a model runs on [MLX](https://github.com/Blaizzy/mlx-audio) (`[mlx]` extra); on CUDA or CPU it runs in its own environment, built automatically the first time.
- **Benchmarked on real speech** (FLEURS, Mac Studio M3 Ultra). Qwen3-ASR 1.7B beats SenseVoice-Small in Mandarin (9.4 vs 10.5% CER), Cantonese (6.6 vs 8.7%) and English (3.9 vs 8.2% WER) while running faster; Fun-ASR-Nano has the lowest Mandarin CER (8.6%). Qwen3-TTS 0.6B synthesizes 5× faster than real time and handles Chinese–English code-switching that Kokoro misreads. Full tables in the [changelog](CHANGELOG.md#210---2026-09-23).
- **Every model in `kin audio models` loads**, and the HTTP server serves all of them. `/synthesize` gains `instruct` for tone and voice design.
- **Fixes found on a clean machine**: SenseVoice no longer runs `pip install` on load, Kokoro can't take the server down when its spaCy setup fails, CosyVoice works for the first time, MusicGen plays at the right pitch.

Recommended:

```bash
kin audio serve qwen3-asr:1.7b --port 8000 --emotion sensevoice:small   # STT, with SenseVoice's emotion labels
kin audio serve qwen3-tts:0.6b --port 8001                              # TTS, accepts Kokoro voice ids
```

See [CHANGELOG.md](CHANGELOG.md) for full history.

## Features

- **STT**: Qwen3-ASR, Fun-ASR-Nano, FireRedASR2, GLM-ASR, Parakeet, Canary, Nemotron, Voxtral Realtime, VibeVoice-ASR, Whisper (openai / faster-whisper / whisper.cpp / MLX), SenseVoice, Paraformer, Moonshine — [full list](#current-models-catalog)
- **TTS**: Qwen3-TTS, VoxCPM2, CosyVoice3, IndexTTS-2, Fish Audio S2 Pro, Chatterbox, Spark-TTS, OmniVoice, CSM, Dia, Higgs Audio, Kokoro, ChatTTS, F5-TTS and more
- **Any platform**: one model name everywhere — MLX on Apple Silicon, torch on CUDA/CPU in auto-built isolated environments
- **Music Generation**: HeartMuLa (multilingual, tag-based), MusicGen
- **Voice Activity Detection**: Engine-agnostic `/vad` endpoint via TEN-VAD; or built-in Silero VAD inside `faster-whisper`
- **Chinese Language Support**: Optimized models for Mandarin, Cantonese, and mixed Chinese-English
- **Voice Cloning**: Zero-shot voice cloning with F5-TTS and CosyVoice
- **MCP Integration**: Use with Claude Code and Claude Desktop
- **WebSocket Streaming**: Real-time transcription and synthesis
- **REST API**: FastAPI-based server with OpenAPI docs; OpenAI-compatible `/v1/audio/transcriptions`
- **Subtitle Output**: Direct SRT / WebVTT generation from `/transcribe?response_format=srt|vtt`

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

# Generate music (with Chinese support!)
kin audio music generate "在月光下弹钢琴"  # Chinese lyrics
kin audio music generate "happy wedding" --tags "piano,romantic,wedding" --model heartmula:3b

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

### Using uv (recommended — 10x faster)

This project has heavy ML dependencies (~4GB: PyTorch, Whisper, transformers). [uv](https://github.com/astral-sh/uv) resolves and installs them **10-100x faster** than pip.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install
uv pip install localkin-service-audio

# Or from source
git clone https://github.com/LocalKinAI/localkin-service-audio.git
cd localkin-service-audio
uv sync
```

**Using in a new terminal:** The virtual environment needs to be activated each session:

```bash
# Option 1: Activate the venv
source .venv/bin/activate
kin audio models

# Option 2: Use uv run (no activation needed)
uv run kin audio models
```

To auto-activate, add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Activate .venv automatically when entering a project directory
cd() { builtin cd "$@" && [ -f .venv/bin/activate ] && source .venv/bin/activate; }
```

### Using pip

```bash
pip install localkin-service-audio
```

> pip works but is significantly slower due to dependency resolution with large ML packages. Expect 10-30 minutes on first install.

### Upgrading

```bash
# Upgrade to latest version
uv pip install --upgrade localkin-service-audio

# If upgrading from v2.0.3 or earlier, also upgrade torch (required for v2.0.4+)
uv pip install --upgrade torch torchaudio torchvision
```

### Optional Dependencies

```bash
# Chinese language models
uv pip install localkin-service-audio[chinese]

# Voice cloning models
uv pip install localkin-service-audio[cloning]

# MCP server for Claude
uv pip install localkin-service-audio[mcp]

# Apple Silicon: run catalog models on MLX (fastest on a Mac). Optional —
# without it they run in auto-built torch environments like everywhere else.
uv pip install "localkin-service-audio[mlx]"

# All features
uv pip install localkin-service-audio[all-new]
```

> Replace `uv pip` with `pip` if not using uv.

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

### Music Generation

```bash
# MusicGen — text-to-music (small/medium/large)
kin audio music generate "calm piano melody"
kin audio music generate "upbeat electronic" --duration 20 --model musicgen:medium
kin audio music generate "ambient soundscape" -o ambient.wav --device mps

# HeartMuLa — multilingual with Chinese lyrics support
kin audio music generate "在月光下弹钢琴" --model heartmula:3b
kin audio music generate "happy wedding day" --tags "piano,romantic,wedding" --model heartmula:3b --duration 30
kin audio music generate "春天来了，鸟儿在唱歌" --tags "acoustic,happy,upbeat" -o spring.wav

# MiniMax Music 3 — full songs with lyrics (MLX on Apple Silicon, else ComfyUI)
kin audio music generate "中文流行抒情，女声，钢琴" --model minimax-music3 --lyrics @song.txt --duration 60 -o song.flac

# Anything ComfyUI has as an audio blueprint: ACE-Step 1.5, YuE2, Stable Audio 3
kin audio music generate "cinematic rain ambience" --model stable-audio3 --comfyui-url http://box:8188
kin audio music generate "..." --model "comfyui:Text to Music (YuE2)" --param cfg_scale=2.0

# List music models and requirements (includes ComfyUI's, with whether the weights are installed)
kin audio music models
kin audio music models --verbose
```

**ComfyUI models** run the blueprint ComfyUI ships for them, on whatever machine runs ComfyUI (`--comfyui-url` or `LOCALKIN_COMFYUI_URL`, default `http://localhost:8188`), using the weights already installed there. `--lyrics` takes text or `@file`; `--param name=value` sets any blueprint input. On Apple Silicon, `minimax-music3` defaults to mlx-audio (`--backend mlx`): through ComfyUI on a Mac its text encoder crawls at several seconds per step.

**HeartMuLa style tags:** `piano`, `acoustic`, `electric`, `synthesizer`, `happy`, `sad`, `romantic`, `calm`, `upbeat`, `wedding`, `ambient`, `orchestral`, `rock`, `pop`, `jazz`, `folk`, `classical`, `cinematic`

| Model | Sizes | VRAM | Languages | Duration |
|-------|-------|------|-----------|----------|
| MusicGen | small (2GB), medium (4GB), large (16GB) | 2–16 GB | English | 5–30s |
| HeartMuLa | 3B (6GB), 7B (16GB) | 6–16 GB | en, zh, ja, ko, es | 5–240s |

**HeartMuLa setup** — auto-installs on first use, or pull in advance:
```bash
kin audio pull heartmula:3b
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
# List all models with availability status
kin audio models

# Filter by type, language, engine, or tag
kin audio models --type stt
kin audio models --type tts
kin audio models --language zh
kin audio models --engine kokoro
kin audio models --tag voice-cloning
kin audio models --search whisper

# Pull a model
kin audio pull whisper-cpp:base
kin audio pull heartmula:3b

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

`kin audio models` shows all 76 models with real-time availability status:
- **✅ Ready** — engine installed, usable now
- **📦 Not installed** — strategy code exists, just needs `pip install`

### Current models (catalog)

Use the same name on any machine. **Mac (MLX)** needs Apple Silicon and the `[mlx]` extra; **torch** runs on CUDA, CPU or MPS in an environment built on first use (needs [uv](https://docs.astral.sh/uv/); set `LOCALKIN_AUTO_INSTALL=0` to refuse, and `kin audio pull <model>` to build ahead of time). Weights download on first use. Repos and parameters live in [`catalog.py`](localkin_service_audio/core/config/catalog.py).

#### Speech-to-Text

| Model | Languages | Notes | Mac (MLX) | CUDA / CPU (torch) |
|---|---|---|---|---|
| `qwen3-asr:0.6b` | zh, en, yue, ja, ko, de … | beats Whisper large-v3 on Chinese, 22 dialects, 30 languages | ✅ | ✅ |
| `qwen3-asr:1.7b` | zh, en, yue, ja, ko, de … | AISHELL-2 WER 2.71 vs 5.06 for Whisper large-v3 | ✅ | ✅ |
| `fireredasr2:aed` | zh, en | lowest Mandarin CER published (AISHELL-1 0.57), 20+ dialects | ✅ | ✅ |
| `fun-asr:nano` | zh, en, ja | Tongyi's ASR, 7 Chinese dialects and 26 accents | ✅ | ✅ |
| `glm-asr:nano` | zh, en, yue | Zhipu's ASR, strong on Cantonese and quiet speech | ✅ | ✅ |
| `confucius4:r2t2` | zh, en | NetEase Youdao's Qwen3-ASR-based recognizer | ✅ | — |
| `moss-transcribe-diarize` | zh, en | speaker labels + timestamps | ✅ | — |
| `whisper-mlx:large-v3-turbo` | en, zh, de, es, fr, ja … | Whisper large-v3-turbo on MLX (elsewhere use faster-whisper:large-v3-turbo) | ✅ | — |
| `nemotron-asr:0.6b` | en, zh, es, de, fr, it … | NVIDIA cache-aware streaming ASR, 40 language-locales | ✅ | ✅ |
| `voxtral-realtime:4b` | en, zh, fr, es, de, it … | Mistral's streaming ASR | ✅ | ✅ |
| `vibevoice-asr:9b` | en, zh, ja, ko, de, fr … | Microsoft, up to 60 min audio with speakers and hotwords | ✅ | ✅ |
| `parakeet:0.6b` | en, fr, de, es, pt, it … | NVIDIA, 25 European languages, far faster than Whisper | ✅ | ✅ |
| `parakeet:0.6b-v2` | en | NVIDIA English ASR (torch: community HF port) | ✅ | ✅ |
| `parakeet:1.1b` | en | NVIDIA English ASR (torch: community HF port) | ✅ | ✅ |
| `canary:1b-v2` | en, fr, de, es, pt, it … | NVIDIA transcription + translation, 25 languages (no auto-detect) | ✅ | ✅ |
| `canary-qwen:2.5b` | en | NVIDIA English ASR leaderboard leader | ✅ | — |

#### Text-to-Speech

| Model | Languages | Notes | Mac (MLX) | CUDA / CPU (torch) |
|---|---|---|---|---|
| `qwen3-tts:0.6b` | zh, en, ja, ko, de, fr … | 9 preset voices incl. dialects, emotion via instruct | ✅ | ✅ |
| `qwen3-tts:1.7b` | zh, en, ja, ko, de, fr … | most-downloaded TTS of 2026, emotion via instruct | ✅ | ✅ |
| `qwen3-tts:1.7b-voicedesign` | zh, en, ja, ko, de, fr … | describe the voice in words (pass instruct) | ✅ | ✅ |
| `qwen3-tts:0.6b-base` | zh, en, ja, ko, de, fr … | 3-second zero-shot voice cloning | ✅ | ✅ |
| `qwen3-tts:1.7b-base` | zh, en, ja, ko, de, fr … | higher-fidelity zero-shot voice cloning | ✅ | ✅ |
| `voxcpm2` | zh, en, ja, ko, de, fr … | Seed-TTS zh CER 0.97, 48 kHz, voice design and 9 Chinese dialects | ✅ | ✅ |
| `cosyvoice3:0.5b` | zh, en, ja, ko, yue, de … | Tongyi, zh CER 1.12, 18+ dialects, instruct control | — | ✅ |
| `cosyvoice2:0.5b` | zh, en, ja, ko, yue, de … | 9 languages + 18 Chinese dialects, voice cloning | — | ✅ |
| `cosyvoice:300m` | zh, en, ja, ko, yue | original CosyVoice with 7 preset voices | — | ✅ |
| `indextts:2` | zh, en | bilibili's zero-shot TTS with emotion control | — | ✅ |
| `indextts:2.5` | zh, en | newer IndexTTS, faster and more stable | — | ✅ |
| `fish-speech:s2-pro` | zh, en, ja, ko, de, fr … | inline emotion tags, voice cloning | ✅ | — |
| `spark-tts:0.5b` | zh, en | gender and pitch control, voice cloning | ✅ | — |
| `longcat-audiodit:1b` | zh, en | Meituan diffusion TTS, zh CER 1.18 | ✅ | — |
| `breeze-tts:2` | zh, en | Taiwanese Mandarin + English, voice design | ✅ | — |
| `confucius4-tts` | zh, en, ja, ko, de, fr … | NetEase Youdao, 14 languages | ✅ | — |
| `moss-tts:nano` | zh, en, ja, ko, de, fr … | tiny multilingual cloning TTS | ✅ | — |
| `moss-tts:local-v1.5` | zh, en, ja, ko, de, fr … | 31 languages | ✅ | — |
| `ming-omni-tts:0.5b` | zh, en | Ant Group, style control | ✅ | — |
| `chatterbox:multilingual` | en, zh, ja, ko, de, fr … | Resemble AI, 23 languages, emotion exaggeration control | ✅ | ✅ |
| `chatterbox:turbo` | en | Resemble AI's fast English TTS with paralinguistic tags | ✅ | ✅ |
| `omnivoice` | zh, en, ja, ko, de, fr … | k2-fsa zero-shot TTS for 600+ languages, nonverbal tags | ✅ | ✅ |
| `higgs-audio:v2` | en, zh, ko, de, es | Boson AI expressive TTS, multi-speaker dialogue | ✅ | ✅ |
| `outetts:1b` | en, zh, ja, ko, de, fr … | Llama-based multilingual TTS | ✅ | — |
| `voxtral-tts:4b` | en, fr, es, de, it, pt … | Mistral, 20 preset voices, no Chinese | ✅ | — |
| `vibevoice:0.5b` | en | Microsoft streaming TTS | ✅ | — |
| `vibevoice:1.5b` | en, zh | Microsoft long-form multi-speaker TTS | — | ✅ |
| `csm:1b` | en | conversational speech model with voice cloning | ✅ | ✅ |
| `dia:1.6b` | en | multi-speaker dialogue in one pass ([S1]/[S2]), laughter and coughs | ✅ | ✅ |
| `orpheus:3b` | en | Llama-based emotional TTS, <laugh> <sigh> tags | ✅ | — |
| `kitten-tts:mini` | en | under 100 MB | ✅ | — |
| `soprano:80m` | en | tiny, very fast English TTS | ✅ | — |
| `pocket-tts` | en, fr, de, pt, it, es | 100M, real-time on CPU, voice cloning | ✅ | — |
| `irodori-tts` | ja | Japanese TTS | ✅ | — |

The torch backends follow each project's documented API. Qwen3-ASR, Qwen3-TTS (MPS) and CosyVoice3 (CPU) have run end to end on the torch path; CUDA runs are awaiting reports — issues welcome. [Benchmark results](CHANGELOG.md#210---2026-09-23) cover 8 STT and 9 TTS models.

### Built-in engines

#### Speech-to-Text

> **Tip — Voice Activity Detection has two paths since v2.0.12:**
>
> 1. **Inline with transcription** — use a `faster-whisper:*` model and pass `enable_vad=true` to `/transcribe`. Silero VAD is bundled inside the engine; transitions are merged into the resulting transcript.
> 2. **Standalone, engine-agnostic** — call `POST /vad` (always available, no model required). Backed by [TEN-VAD](https://huggingface.co/TEN-framework/ten-vad) — 731 KB native macOS arm64 binary, faster transitions than Silero. Returns raw speech segments so you can chunk audio before transcription or use it for diarization-lite workflows. See the [/vad endpoint docs](#endpoints).

| Model | Engine | Languages | Features | Status |
|-------|--------|-----------|----------|--------|
| `whisper:tiny/base/small/medium/large-v3` | OpenAI Whisper | Multilingual | Standard reference | Ready |
| `whisper:large-v3-turbo` | OpenAI Whisper | Multilingual | 6x faster than large-v3, 809M params | Ready |
| `faster-whisper:tiny/base/large-v3/turbo/distil-large-v3` | CTranslate2 | Multilingual | 4x faster, GPU, **native VAD** | Ready |
| `faster-whisper:large-v3-turbo` | CTranslate2 | Multilingual | CTranslate2 turbo variant | Ready |
| `whisper-cpp:tiny/base/small/medium` | whisper.cpp | Multilingual | Fast CPU inference | Ready |
| `moonshine:tiny/base` | Moonshine | English | 5x real-time, ~20MB | Install needed |
| `sensevoice:small` | FunASR (Alibaba) | zh, en, ja, ko | 15x faster, emotion detection | Install needed |
| `paraformer:zh` | FunASR (Alibaba) | Chinese | Fast Chinese ASR | Install needed |

#### Text-to-Speech

| Model | Engine | Languages | Features | Status |
|-------|--------|-----------|----------|--------|
| `native` | pyttsx3 | System | No download needed | Ready |
| `kokoro` / `kokoro:82m` | Kokoro | en, es, fr, hi, it, ja, pt, zh | 54 voices, multilingual | Ready |
| `chattts` | ChatTTS | zh, en | Conversational, emotion | Install needed |
| `f5-tts` | F5-TTS | en, zh | Zero-shot voice cloning | Install needed |

### Music Models

| Model | Engine | Languages | Features | Status |
|-------|--------|-----------|----------|--------|
| `musicgen:small/medium/large` | MusicGen (Meta) | English | Text-to-music, 5–30s | Install needed |
| `heartmula:3b/7b` | HeartMuLa | en, zh, ja, ko, es | Chinese lyrics, tag control, up to 240s | Install needed |

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
# Basic (language is a query parameter; omit it to auto-detect)
curl -X POST "http://localhost:8000/transcribe?language=en" \
  -F "file=@audio.wav"
```

With SenseVoice as the model, or any model served with `--emotion sensevoice:small`, the JSON also carries `emotion` (`happy` / `sad` / `angry` / `neutral` / …) and `audio_events` (laughter, applause, …). Treat the label as a hint.

Optional query parameters (added in v2.0.11):

| Param | Type | Default | Notes |
|---|---|---|---|
| `language` | string | auto | BCP-47 language code, e.g. `en`, `zh` |
| `enable_vad` | bool | `true` | Skip silence via VAD (faster-whisper only) |
| `timestamps` | bool | `false` | Include segment timings in JSON response |
| `response_format` | enum | `json` | `json` \| `text` \| `markdown` \| `srt` \| `vtt` |
| `chunk_length_s` | int | engine default | VRAM tuning for long audio |

```bash
# Markdown transcript with timestamps
curl -X POST 'http://localhost:8000/transcribe?response_format=markdown' \
  -F 'file=@meeting.wav'

# SRT subtitles for video captioning
curl -X POST 'http://localhost:8000/transcribe?response_format=srt' \
  -F 'file=@video.wav' > captions.srt

# Low-VRAM long-audio: VAD + smaller chunks
curl -X POST 'http://localhost:8000/transcribe?chunk_length_s=15&enable_vad=true' \
  -F 'file=@long.wav'

# JSON with segment timestamps (no shape change to existing callers
# unless you opt in with timestamps=true)
curl -X POST 'http://localhost:8000/transcribe?timestamps=true' \
  -F 'file=@audio.wav'
```

**POST /synthesize** - Synthesize speech (returns `audio/wav`)
```bash
curl -X POST "http://localhost:8001/synthesize" \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天气很好", "speaker": "vivian", "instruct": "用开心的语气"}' \
  --output speech.wav
```

| Field | Default | Notes |
|---|---|---|
| `text` | — | Required |
| `speaker` | model default | A voice id from `GET /voices`. Kokoro ids (`zf_xiaoxiao`, `af_heart`…) are mapped to the nearest voice on other models |
| `language` | from the text | ISO code, e.g. `zh`, `en` |
| `speed` | `1.0` | 0.5–2.0 |
| `instruct` | — | Tone on Qwen3-TTS CustomVoice ("用开心的语气"), a voice description on VoiceDesign models; ignored elsewhere |

**GET /voices** - Voices of the TTS model
```bash
curl "http://localhost:8001/voices"
# {"model": "qwen3-tts:0.6b", "multilingual": true, "default_voice": "vivian",
#  "voices": [{"id": "vivian", "name": "Vivian · 明亮女声", "language": "zh", "gender": "female"}, ...]}
```

`multilingual: true` means every voice reads every language the model supports, so send mixed-language text in one request rather than splitting it between voices (which single-language Kokoro voices need).

**POST /vad** - Detect speech segments (no transcription)

Engine-agnostic Voice Activity Detection via [TEN-VAD](https://huggingface.co/TEN-framework/ten-vad)
(731 KB native macOS arm64 binary, ~0.016 RTF on M1, ~100-300 ms faster
transitions than Silero). Useful for chunking long audio before
transcription, or for VAD-only workflows.

```bash
# Install the optional VAD extra first
pip install 'localkin-service-audio[vad]'

curl -X POST "http://localhost:8000/vad" \
  -F "file=@meeting.wav"

# Output:
# {
#   "backend": "ten-vad",
#   "duration": 132.4,
#   "speech_segments": [
#     {"start": 1.2, "end": 5.8, "duration": 4.6},
#     ...
#   ],
#   "total_speech_duration": 48.3
# }
```

Tunable parameters (all optional query strings):

| Param | Default | Effect |
|---|---|---|
| `backend` | `ten-vad` | VAD backend (currently only one supported) |
| `threshold` | `0.5` | 0.0-1.0 speech-probability cutoff |
| `min_speech_duration_ms` | `200` | Drop speech runs shorter than this |
| `min_silence_duration_ms` | `200` | Merge runs separated by less silence |
| `speech_pad_ms` | `100` | Pad each kept segment by this much |

**GET /models** - List models
```bash
curl "http://localhost:8000/models"
```

**GET /health** - Readiness. `503` with the missing package when the model's backend isn't installed; `loaded: true` once the model is in memory (`serve` preloads by default).

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

# Install with dev dependencies (uv recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
black --check .
```

> **Tip:** With `uv`, you can skip activation and run commands directly:
> ```bash
> uv run kin audio models
> uv run pytest tests/
> ```

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

### PyTorch Version

Requires **torch >= 2.6.0**. Older versions will fail to load models that only ship `.bin` weights (e.g. MusicGen medium/large) due to a `torch.load` security check (CVE-2025-32434).

```bash
# Check your version
python -c "import torch; print(torch.__version__)"

# Upgrade if needed (keep torchvision in sync)
pip install "torch>=2.6.0" "torchaudio>=2.6.0" "torchvision>=0.21"
```

### macOS 27: `__thread_bss` / scipy import error

scipy 1.15.3, the last release with Python 3.10 wheels, fails to load on macOS 27. Create the environment with Python 3.11+:

```bash
uv venv -p 3.11 && uv pip install -e ".[sensevoice]"
```

### numpy/pandas Binary Incompatibility

If you see `numpy.dtype size changed, may indicate binary incompatibility`, pandas or scikit-learn was compiled against a different numpy version:

```bash
# Fix: force-reinstall the affected packages
uv pip install --force-reinstall numpy pandas scikit-learn

# Or nuke and rebuild the venv
rm -rf .venv && uv venv && uv pip install localkin-service-audio
```

### CUDA/GPU Issues

```bash
# Force CPU
kin audio transcribe audio.wav --device cpu

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### HeartMuLa on Apple Silicon (MPS)

HeartMuLa 3B requires ~12-14GB. On a 16GB Mac, close memory-heavy apps before running. The codec runs on CPU automatically (shared unified memory, no performance impact).

```bash
# If you hit OOM, try shorter duration
kin audio music generate "prompt" --model heartmula:3b --duration 5

# Or force CPU (slower but more stable memory management)
kin audio music generate "prompt" --model heartmula:3b --device cpu
```

### Chinese Model Dependencies

SenseVoice and Paraformer run on FunASR, which is an optional extra — it pulls
in modelscope and a sizeable tree, so a Whisper-only install shouldn't have to
download it.

```bash
# Install the extra (declared in pyproject.toml)
uv pip install --project . -e ".[sensevoice]"

# Or plain pip
pip install funasr modelscope

# Then use Chinese models
kin audio transcribe audio.wav --model sensevoice:small

# Or serve them
kin audio serve sensevoice:small --port 8000
```

> Without the extra, the server still starts and `/health` still reports
> `{"status": "healthy"}` — the failure only shows up on the first
> `/transcribe`, as `funasr not installed`. The model weights are downloaded
> separately and being present is not enough on its own.

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [MusicGen](https://github.com/facebookresearch/audiocraft)
- [HeartMuLa](https://github.com/HeartMuLa/heartlib)
