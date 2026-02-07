---
name: localkin-audio
description: "Local Speech-to-Text and Text-to-Speech via LocalKin Audio. Transcribe audio files, synthesize speech, manage models, and run real-time voice conversations — all locally, no API keys needed."
homepage: https://github.com/localkin/localkin-service-audio
user-invocable: true
metadata: {"openclaw":{"emoji":"🎵","os":["darwin","linux"],"requires":{"bins":["kin"]},"install":[{"id":"uv","kind":"uv","package":"localkin-service-audio","bins":["kin"],"label":"Install via uv"},{"id":"pip","kind":"pip","package":"localkin-service-audio","bins":["kin"],"label":"Install via pip"}]}}
---

# LocalKin Audio

Local voice AI platform for Speech-to-Text (STT) and Text-to-Speech (TTS). All processing runs locally on the user's machine — no cloud API keys required.

## When to use this skill

Use this skill when the user wants to:

- **Transcribe** an audio file to text (speech-to-text)
- **Synthesize** speech from text (text-to-speech)
- **Listen** in real-time from a microphone with optional LLM conversation
- **Manage models** — list, download, remove, or benchmark STT/TTS models
- **Check system status** — hardware detection, installed engines, cache info

## Quick reference

### Transcribe audio to text

```bash
kin audio transcribe <audio_file>
kin audio transcribe recording.wav --model whisper-cpp:base --language en
kin audio transcribe interview.mp3 --format srt --timestamps --output subtitles.srt
kin audio transcribe meeting.wav --format json --output transcript.json
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m` | STT model (engine:variant) | `whisper-cpp:base` |
| `--language, -l` | Language code (auto-detect if omitted) | auto |
| `--output, -o` | Output file (stdout if omitted) | stdout |
| `--format, -f` | Output format: `text`, `json`, `srt`, `vtt` | `text` |
| `--timestamps` | Include timestamps | off |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--verbose, -v` | Verbose output | off |

### Synthesize speech (TTS)

```bash
kin audio tts "Hello, world!"
kin audio tts "Welcome to LocalKin" --voice af_heart --output welcome.wav
kin audio tts "你好世界" --model kokoro --voice zf_xiaobei
kin audio tts "Bonjour" --speed 0.8 --no-play --output bonjour.wav
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m` | TTS model | `kokoro` |
| `--voice, -v` | Voice ID | model default |
| `--output, -o` | Output file (plays directly if omitted) | play |
| `--speed, -s` | Speed multiplier | `1.0` |
| `--play/--no-play` | Play audio after synthesis | `--play` |
| `--list-voices` | List available voices and exit | — |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` |

**Kokoro voice IDs follow a pattern:** first letter = language (a/b=English, e=Spanish, f=French, h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese), second letter = gender (f=female, m=male). Examples: `af_heart` (English female), `zm_yunyang` (Chinese male), `jf_alpha` (Japanese female).

To list all available voices: `kin audio tts --list-voices`

### Real-time listening

```bash
kin audio listen
kin audio listen --tts --tts-voice af_heart
kin audio listen --llm ollama --llm-model qwen3:14b --tts --stream
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--model, -m` | STT model | `whisper-cpp:base` |
| `--tts/--no-tts` | Enable TTS responses | off |
| `--tts-model` | TTS model for responses | `native` |
| `--tts-voice` | Voice for TTS | — |
| `--llm` | LLM backend (e.g. `ollama`) | off |
| `--llm-model` | LLM model name | `qwen3:14b` |
| `--stream/--no-stream` | Stream LLM responses | off |
| `--language, -l` | Language code | auto |
| `--silence-threshold` | Silence threshold 0.0–1.0 | `0.01` |
| `--silence-duration` | Seconds of silence before processing | `1.5` |

### Model management

```bash
# List all available models
kin audio models

# Filter models
kin audio models --type stt
kin audio models --type tts --language zh
kin audio models --engine kokoro
kin audio models --search "chinese"

# Download a model
kin audio pull whisper-cpp:base
kin audio pull kokoro

# Remove a model
kin audio rm whisper-cpp:base

# Get hardware-based recommendations
kin audio recommend
kin audio recommend --verbose
```

### Benchmark models

```bash
kin audio benchmark test.wav
kin audio benchmark test.wav --models whisper-cpp:tiny whisper-cpp:base faster-whisper:base
kin audio benchmark test.wav --iterations 3 --output results.json
```

### System and configuration

```bash
# Check system status (installed engines, libraries, cache)
kin audio status

# View configuration
kin audio config
kin audio config --path
kin audio config --models

# Set configuration
kin audio config set default_stt_model whisper-cpp:base
kin audio config set default_tts_model kokoro
kin audio config set default_device mps

# Cache management
kin audio cache info
kin audio cache clear
kin audio cache clear whisper-cpp:base

# Show running servers
kin audio ps
```

### API server

```bash
# Start the API server
kin audio serve
kin audio serve --host 0.0.0.0 --port 8000
kin audio serve whisper-cpp:base  # pre-load a model

# Start web UI
kin web
kin web --port 8080
```

### Add custom models

```bash
# List available templates
kin audio list-templates

# Add from template
kin audio add-model --template whisper-gguf --name my-whisper --repo user/model

# Add from Hugging Face
kin audio add-model --name my-tts --repo user/tts-model --type tts --size-mb 200
```

## Important notes

- **Model format:** Models are specified as `engine:variant` (e.g., `whisper-cpp:base`, `faster-whisper:large-v3`). Some models like `kokoro` have no variant.
- **First run:** Models are downloaded on first use. Use `kin audio pull <model>` to pre-download.
- **Device selection:** Use `--device mps` on Apple Silicon Macs for GPU acceleration. The default `auto` usually picks the best option.
- **Output files:** When `--output` is omitted, transcriptions print to stdout and TTS plays audio directly.
- **Supported audio formats:** WAV, MP3, FLAC, OGG, M4A, and other formats supported by ffmpeg.

## Scripted usage

For scripted/automated use, helper scripts are available at `{baseDir}/scripts/`:

- `{baseDir}/scripts/transcribe.sh <audio_file> [options]` — transcribe wrapper
- `{baseDir}/scripts/tts.sh "<text>" [options]` — TTS wrapper
- `{baseDir}/scripts/status.sh` — quick status check
