# Changelog

All notable changes to LocalKin Service Audio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.6] - 2026-10-12

Windows reads and writes its text files as UTF-8.

### Fixed

- **Text files are UTF-8 on Windows.** The custom models config, the settings file and an isolated environment's requirements were read in the locale's encoding, which is cp1252 on Windows. Chinese in them came out garbled or stopped the read.
  - Every text file the package reads or writes now names UTF-8, and a test keeps it that way.
  - Found by the first Windows CI run, where a test fixture wrote Chinese the same way.

## [2.1.5] - 2026-10-12

Emotion at no extra wait, failures that say why, and tests on every push.

### Changed

- **Emotion runs alongside transcription.** With `--emotion`, SenseVoice now listens while the main model transcribes, instead of afterwards.
  - On a Mac Studio with Qwen3-ASR 1.7B, a 2 s clip took 0.55 s instead of 0.66 s, 8 s took 0.61 s instead of 0.82 s, and 18 s took 0.72 s instead of 1.08 s.
  - Transcription no longer blocks the server while it runs. Each model still takes one request at a time.
- **Logs say why.** Every 500 from the server now logs its traceback; three transcription failures on a box had left no cause. Each transcription, synthesis and stream logs one line with how long it took.
- **`kin audio recommend` on NVIDIA and CPU** suggests current models: Qwen3-ASR, Qwen3-TTS, CosyVoice3, Parakeet and SenseVoice.
  - It used to suggest Whisper large-v3, CosyVoice 1, F5-TTS and ChatTTS.
  - A test keeps Apple-only models off those lists.
- **Dependencies.** The lockfile takes the fixed releases for 143 of 146 Dependabot alerts: aiohttp, anyio, cryptography, nltk, pillow, starlette and fastapi, torch, urllib3 and others.
  - The other three are nltk advisories with no fixed release yet. nltk only comes with the `tts` extra (Coqui TTS).

### Added

- **CI.** GitHub Actions runs the unit tests from the lockfile on Linux, macOS and Windows with Python 3.11, and on Linux with Python 3.10.

## [2.1.4] - 2026-10-10

Agents on other machines, and speech that starts before it's finished.

### Added — agents on other machines, streaming, access keys

- **Remote MCP.** `kin mcp --http` serves MCP over streamable HTTP (at `/mcp`) for agents on other machines or in the cloud.
  - Audio comes in as a URL or base64. Results go back as a download URL (`/files/...`) and, on request, as base64.
  - Paths only reach the server's own results, and `output_path` isn't accepted.
  - `--public-url` sets the address used in links, for use behind proxies and tunnels.
  - Tested from another machine with the MCP SDK's client against real models.
- **Access keys.** `--api-key` (or `LOCALKIN_API_KEY`) on `kin audio serve` and `kin mcp --http` requires the key.
  - Clients send it as `Authorization: Bearer <key>`, which is what an OpenAI SDK sends as its `api_key`, or as `X-API-Key`.
  - `/health` and the docs stay open. WebSocket routes are covered too.
- **Streaming speech.** `/v1/audio/speech` takes OpenAI's `stream_format` (`audio` or `sse`) with pcm, wav or mp3.
  - pcm streams by default, as OpenAI's does.
  - Qwen3-TTS on MLX streams within a sentence, a piece every 0.6 s of speech. Other models go one sentence at a time.
  - On a Mac Studio, a 23-second reply started after 0.17 s instead of 4.8 s.
- **MCP tool inputs and outputs.** Over stdio as well as HTTP, the tools now also take audio as `audio_url`/`audio_base64` (`reference_url`/`reference_base64` for cloning), and results take `return_base64`.
- **README** setup for VS Code, Windsurf and Gemini CLI.

### Changed

- **One synthesis at a time per model** in the HTTP server, because streams run in their own thread. Waiting requests wait off the event loop, so the server stays responsive.

## [2.1.3] - 2026-10-10

Connecting it to an agent now works from a plain `pip install`.

### Added — agents

- **The MCP server (`kin mcp`) works.** It didn't before, for three reasons:
  - the `[mcp]` extra didn't install the MCP SDK;
  - `kin mcp` printed a banner to stdout, the protocol's channel, which breaks clients;
  - the 2.x SDK, now the default, dropped the API it was written against.

  What it does now:
  - It runs on the SDK's 1.x and 2.x. stdout carries only the protocol: model libraries' output, Python or native, goes to stderr.
  - Default models are `qwen3-asr` (1.7B, or 0.6B under 16 GB of RAM), `qwen3-tts:0.6b`, a `qwen3-tts` base model for cloning, and `musicgen:small`. Change them with `--stt`, `--tts`, `--clone` and `--music`, or with `LOCALKIN_*_MODEL`. Speech, cloning and transcription each keep their own model loaded.
  - Tools: `transcribe_audio`, `synthesize_speech` (voice, `instruct` for tone, language, speed), `clone_voice`, `generate_music` (new), `list_models`, `list_voices`, `get_audio_info`.
  - The README covers setup for Claude Code, Claude Desktop, Cursor and Codex.
  - Tested with the SDK's own client against real models.
- **`kin audio serve --stt MODEL --tts MODEL`** serves both on one port, so an OpenAI client or agent framework needs a single base URL.
- **OpenAI compatibility.**
  - `/v1/audio/transcriptions` reads the form fields OpenAI SDKs send (`language`, `response_format`); query parameters still work. It can also return `verbose_json`.
  - `/v1/audio/speech` takes `instructions`, and `response_format` `mp3`, `opus`, `flac` or `pcm`. `wav` stays the default.
  - `/v1/models` lists the models.
  - Checked with the official OpenAI Python SDK.
- **Agent skill.** `kin skill` installs a skill that teaches Claude Code (or, with `--dir`, another agent) which model fits which job and the commands for it. The OpenClaw skill is the same text, and a test keeps every option it uses real.
- **New `kin audio tts` options:** `--instruct` for tone, `--language`, and `--ref-audio`/`--ref-text` for cloning. `--list-voices` no longer needs a text.

### Fixed

- **Uploads to `/transcribe` and `/v1/audio/transcriptions` keep their format.** An mp3 used to be saved as `.wav` and failed to decode. Audio that isn't WAV is now converted: mp3, flac and ogg directly, m4a and webm through ffmpeg. The same applies to files given to `kin audio transcribe` and to the MCP tools.
- **Load failures exit non-zero.** `kin audio transcribe` and `kin audio tts` now print the reason a model failed to load and exit with an error. Transcription's stdout carries only the transcript, with loading messages on stderr, so `-f json` parses.
- **MusicGen runs on the CPU on Apple Silicon.** On MPS its sampling produced NaN and failed.
- **The lockfile pins scipy 1.17 on Python 3.11+.** The 1.15.3 it had doesn't load on macOS 27, so `uv sync` installs broke there.

## [2.1.2] - 2026-10-10

### Added — `/synthesize` clones a voice

- **`reference_audio`** (a few seconds of someone talking, as base64 WAV or a
  path on the server's machine) and **`reference_text`** (what it says) on
  `/synthesize`: a voice-cloning model (`kin audio serve qwen3-tts:1.7b-base`)
  says the text in that voice, in any language it speaks. A model that cannot
  clone answers 400 instead of ignoring the recording; a base64 recording is
  written to a temporary file and removed after use. KinClaw's 说话视频 and
  OpenMontage's `video_translate` use it, served on its own port (8103) beside
  the voice-design narrator.

### Added — three models

- **`higgs-audio:v3`**: Higgs TTS 3 (Boson AI, 4B), made for voice chat.
  Inline tags control emotion, pauses and sounds; it covers 100+ languages
  and clones voices. Apple Silicon (mlx-audio).
- **`audio8-tts:0.6b`**: Audio8 TTS Preview 0.6B. It speaks Mandarin,
  Cantonese and nine more languages, clones voices, and outputs 44.1 kHz.
  Apple Silicon (mlx-audio). Neither TTS model has preset voices, so a plain
  request gets the bundled reference voice.
- **`audio8-asr:infinite`**: Audio8 ASR Infinite (Edge0, 4B), streaming
  Chinese/English recognition.
  - It runs through its official package in its own environment (torch 2.13,
    transformers 5.13) on CUDA, CPU or MPS.
  - Without a language it transcribes as Chinese.
  - No MLX backend yet: mlx-audio doesn't include the model, and the
    community MLX port loads model code shipped inside its own repo.

Measured on the Mac Studio with the 2.1.0 benchmark (the same FLEURS
slices and five Chinese sentences). Speed is processing time divided by
audio length.

| Model | Result | Speed |
|---|---|---|
| `higgs-audio:v3` | all five sentences word-perfect | 0.39 |
| `audio8-tts:0.6b` | all five sentences word-perfect, 44.1 kHz | 0.43 |
| `audio8-asr:infinite` (torch on MPS) | Mandarin CER 10.9, English WER 15.3 when told `en`, no Cantonese | 0.69 |

On the same run `qwen3-tts:0.6b` scored word-perfect at 0.19, and
`qwen3-asr:1.7b` scored 9.4 Mandarin and 3.9 English. Both stay the
recommendations. Audio8 ASR's strength is streaming with end-of-turn
detection, which needs a streaming interface the server doesn't have yet.

## [2.1.1] - 2026-09-24

### Added — music through ComfyUI and MLX

- **`kin audio music generate --model minimax-music3|ace-step:1.5|yue2|stable-audio3`**
  runs the audio blueprints ComfyUI ships, on a local or remote ComfyUI
  (`--comfyui-url`, `LOCALKIN_COMFYUI_URL`), reusing the weights installed
  there. The blueprint — a UI subgraph that only ComfyUI's front end could run
  — is converted to an API graph using ComfyUI's own `/object_info`, so
  `comfyui:<any blueprint name>` works too. Missing model files are named
  before anything is queued; an interrupted run says so.
- **`minimax-music3` on MLX** (`mlx-community/MiniMax-Music3-mxfp8`),
  the default on Apple Silicon: 29 s of song in 188 s on a Mac Studio M3
  Ultra. Through ComfyUI on the same machine, the model's text encoder ran
  on the CPU — where ComfyUI puts every text encoder on a Mac, and its int8
  weights can't use the Apple GPU anyway — and its autoregressive stage took
  3–9 s per step: about an hour for 30 seconds of music.
- `--lyrics` (text or `@file`), `--seed`, `--param name=value`,
  `--backend auto|mlx|comfyui`; `--duration` now defaults to each model's own.
  `kin audio music models` prints the same table as `kin audio models`,
  including ComfyUI's audio blueprints and whether their weights are
  installed. Disabled dead-end nodes in a blueprint (a bypassed preview) are
  dropped rather than refused.

## [2.1.0] - 2026-09-23

50 current open models — the ones people actually download and star — under
one name each, runnable on a Mac (MLX) or anywhere else (torch, in an
environment built on first use). Every model the CLI lists now loads, the HTTP
server serves all of them, and a benchmark on real speech picked the new
defaults. It also fixes a long list of bugs found by running the project on a
clean machine instead of a developer's.

### Benchmark (Mac Studio M3 Ultra, MLX)

Speech-to-text on FLEURS dev — real read speech — 60 Mandarin, 30 Cantonese
and 30 English utterances; CER for Chinese, WER for English, numbers
normalised to characters before scoring. Speed is processing time / audio
length.

| Model | Mandarin | Cantonese | English | Speed |
|---|---|---|---|---|
| `fun-asr:nano` | **8.6** | 8.0 | 5.8 | **0.012** |
| `confucius4:r2t2` | 9.2 | 7.0 | 4.4 | 0.018 |
| `fireredasr2:aed` | 9.2 | 9.3 | 11.0 | 0.077 |
| `qwen3-asr:1.7b` | 9.4 | **6.6** | **3.9** | 0.018 |
| `glm-asr:nano` | 9.5 | 17.4 | 8.0 | 0.019 |
| `qwen3-asr:0.6b` | 9.6 | 7.9 | 4.6 | 0.012 |
| `sensevoice:small` | 10.5 | 8.7 | 8.2 | 0.05 |
| `whisper-mlx:large-v3-turbo` | 11.1 | 12.4 | 5.3 | 0.036 |

Text-to-speech: five Chinese sentences (everyday speech, polyphones and
numbers, Chinese–English code-switching, dates, a long sentence).
Transcribing the output back with `qwen3-asr:1.7b` found it word-perfect
except where noted.

| Model | Speed | Output | Notes |
|---|---|---|---|
| `qwen3-tts:0.6b` | 0.19 | 24 kHz | preset voices, `instruct` for tone |
| `qwen3-tts:1.7b` | 0.21 | 24 kHz | |
| `spark-tts:0.5b` | 0.21 | 16 kHz | one character wrong on code-switching |
| `longcat-audiodit:1b` | 0.40 | 24 kHz | one character wrong on the date |
| `voxcpm2` | 0.48 | 48 kHz | |
| `fish-speech:s2-pro` | 0.52 | 44.1 kHz | |
| `kokoro` | 0.06 | 24 kHz | **35% CER on the iPhone/email sentence** — its Chinese voice can't say English words |
| `cosyvoice3:0.5b` | 2.1 | 24 kHz | torch backend on CPU — CosyVoice has no MPS path |

Recommended defaults: **`qwen3-asr:1.7b`** (best overall, `fun-asr:nano` if
only Mandarin matters) and **`qwen3-tts:0.6b`**. Keep `sensevoice:small` where
you need its emotion and audio-event labels.

### Added

- **Model catalog** (`core/config/catalog.py`): 16 STT and 34 TTS models, each
  with an `mlx` and/or `torch` backend.
  - STT: Qwen3-ASR 0.6B/1.7B, Fun-ASR-Nano, FireRedASR2, GLM-ASR-Nano,
    Confucius4-R2T2, MOSS-Transcribe-Diarize, Nemotron 3.5, Voxtral Realtime,
    VibeVoice-ASR, Parakeet v3/v2/1.1B, Canary 1B v2, Canary-Qwen, Whisper on MLX.
  - TTS: Qwen3-TTS (CustomVoice 0.6B/1.7B, VoiceDesign, Base 0.6B/1.7B),
    VoxCPM2, CosyVoice3/2/300M, IndexTTS-2/2.5, Fish Audio S2 Pro, Spark-TTS,
    LongCat-AudioDiT, Breeze-TTS-2, Confucius4-TTS, MOSS-TTS, Ming-Omni,
    Chatterbox, OmniVoice, OuteTTS, Higgs Audio v2, Voxtral TTS, VibeVoice,
    CSM, Dia, Orpheus, KittenTTS, Soprano, Pocket TTS, Irodori.
- **Backend selection** (`core/config/backends.py`): mlx on Apple Silicon when
  mlx-audio is installed, torch otherwise. `LOCALKIN_BACKEND=mlx|torch` forces
  one.
- **mlx backend**: one STT and one TTS strategy drive every family
  mlx-audio 0.5.5 ships (`[mlx]` extra). `generate` kwargs are filtered against
  each model's signature, and language is spelled the way each model wants.
- **torch backend in isolated environments**: the official packages can't
  share one — qwen-tts pins transformers 4.57, chatterbox-tts torch 2.6 and
  numpy<2, IndexTTS torch 2.8, CosyVoice has no package, FireRedASR2 needs
  Python 3.11 — so each family runs as a worker in its own uv-built venv under
  `~/.localkin-service-audio/envs/`, created on first use.
  `LOCALKIN_AUTO_INSTALL=0` refuses to build; `kin audio pull <model>` builds
  ahead of time.
- **Bundled reference voices** (`assets/voices/`, Kokoro-generated,
  Apache-2.0): models without preset voices clone one when the caller gives no
  `ref_audio`, so the voice is the same on every call.
- `POST /synthesize` takes `instruct` — tone ("用开心的语气") on Qwen3-TTS
  CustomVoice, a voice description on the VoiceDesign models.
- **Emotion sidecar**: `kin audio serve qwen3-asr:1.7b --emotion sensevoice:small`
  takes the transcript from Qwen3-ASR and `emotion` / `audio_events` from
  SenseVoice, so clients that read the emotion keep working with the more
  accurate recogniser. A sidecar failure never fails the transcription.
- **`serve` preloads and warms up** the model before accepting requests
  (`--no-preload` for the old lazy behaviour). It used to print "Pre-loading"
  and load nothing, so the first request after a restart waited 1–2 minutes
  for Qwen3 to load — past a 15 s client timeout — and then ~3 s more for MLX
  kernel compilation; it now answers in under a second.
- **`GET /voices`** on TTS servers lists the model's voices (id, display name,
  native language, gender) and whether they are `multilingual` — each reads
  every language the model supports, so clients shouldn't split mixed-language
  text between voices as single-language Kokoro voices require. Qwen3-TTS
  reports its nine speakers, including the Beijing (Dylan) and Sichuan (Eric)
  dialect voices.
- **Kokoro voice ids work on Qwen3-TTS**: `zf_xiaoxiao`, `zm_yunxi`,
  `af_heart`... map by language and gender to the nearest Qwen3 speaker, and
  unknown names (OpenAI's `alloy`) fall back to the default voice, so a client
  written for Kokoro can switch models without changes.
- `/transcribe` returns SenseVoice's `emotion` and `audio_events` in JSON.
  Treat the label as a hint: a tired-sounding synthetic "今天好累" came back
  `happy`.
- `POST /synthesize` honours `speed` (0.5–2.0); `/v1/audio/speech` maps its
  own `speed` onto it. Kokoro renders at the rate; the catalog models are
  time-stretched after synthesis (WSOLA, pitch unchanged), because most ignore
  a speed argument — Qwen3-TTS takes one on both MLX and torch and does
  nothing with it.

### Fixed

- **Twelve listed models never loaded.** Parakeet, Canary, Canary-Qwen,
  Orpheus, Qwen3-TTS, Dia, Parler and GPT-SoVITS had no implementation and
  failed with `Unknown engine`; they are catalog entries that run now (see
  Removed for the rest). Tests fail if any registry entry lacks a strategy or
  `kin audio recommend` suggests one that can't load — it suggested
  `canary:1b` and `gpt-sovits`.
- **`AudioEngine` ignored the registry.** The engine came from the name's
  prefix, so `cosyvoice2:0.5b` looked for a `cosyvoice2` engine. Names now
  resolve through the registry, which also supplies repo and parameters.
- **The HTTP server served only a few engines.** CosyVoice, ChatTTS and F5-TTS
  had no route and Paraformer fell through to a transformers pipeline that
  can't load it; any model with a strategy is now served.
- **CosyVoice never loaded**: wrong import, no v2/v3 support, a hard-coded
  22.05 kHz rate (v2/v3 are 24 kHz), the v1-only instruct API. It now runs from
  the official repo's `AutoModel` in an isolated environment.
- **SenseVoice ran `pip install` on load.** `trust_remote_code=True` made
  funasr `pip install -r` the model repo's requirements (`numpy<=1.26.4`,
  `gradio`) into whatever pip was on PATH — downgrading numpy, or failing
  outright in a uv venv. SenseVoice is built into funasr, so no remote code.
- **SenseVoice started "healthy" and failed every request**: funasr and
  modelscope were never declared. Added the `[sensevoice]` extra, and
  `/health` now returns 503 naming the missing backend package (or `uv`, for
  isolated models).
- **Kokoro read Chinese in an English voice.** With no speaker it defaulted
  to `af_heart`, which reads Chinese as "Chinese letter, Chinese letter…" or
  returns silence. It now picks a voice from the text's script (kana before
  Han), fails loudly on empty audio, and `/synthesize` goes through the same
  code. A first version of this fix never took effect, because `load()` still
  pinned `af_heart`.
- **Kokoro could take the server down.** Its one-time spaCy model install
  used a bare `uv pip install` (wrong venv), and when neither uv nor pip was
  reachable — a uv venv run from a service or `nohup` — `spacy download`
  called `sys.exit()`. It now installs into the running interpreter, falls
  back to unpacking the wheel, and a failed setup is a load error, not an exit.
- **Isolated environments couldn't find uv** from services and cron jobs;
  it's also looked for in `~/.local/bin`, `~/.cargo/bin` and Homebrew.
- **MusicGen clips played at half speed, an octave low** — labelled 16 kHz,
  decoded at 32 kHz. The test that should have caught it turned every
  exception, assertion failures included, into a skip.
- **SpeechT5 changed voice on every request**; the speaker embedding is now
  drawn once from a fixed seed.
- Synthesis errors are no longer re-wrapped as "Model loading failed: 500: …".
- `uv lock` failed building openai-whisper (its sdist imports
  `pkg_resources`, removed in setuptools 81); build dependencies are capped.

### Changed

- The lockfile now resolves transformers 5.17 and numpy 2.2 (was 4.57 and
  1.22). A fresh `uv sync --frozen` with the `sensevoice`, `vad`, `web` and
  `dev` extras on Python 3.10 passes the full suite and real Kokoro, SenseVoice
  and faster-whisper runs. The legacy `tts` extra (Coqui pins numpy 1.22) is
  declared as conflicting with `mlx`, so it resolves separately.
- `fun-asr:nano` uses `mlx-community/Fun-ASR-Nano-2512`; the quantised builds
  predate the tokenizer layout mlx-audio 0.5.5 loads. IndexTTS and CosyVoice
  are torch-only: no MLX build of them loads in mlx-audio 0.5.5.
- `kin audio recommend` suggests Qwen3-ASR/TTS on Apple Silicon.
- The MCP `clone_voice` tool defaults to `cosyvoice3:0.5b` (the 300M SFT
  build it used can't clone).

### Removed

- `orpheus:1b` (the repo doesn't exist), `orpheus:150m` (pointed at the 3B
  model), `parler-tts` and `gpt-sovits` (no implementation).
- `CosyVoiceStrategy`, replaced by the isolated CosyVoice worker.

### Known issues

- **Python 3.10 on macOS 27**: scipy's last 3.10 wheel doesn't load there
  (`__thread_bss` error). Use Python 3.11+.
- CUDA hasn't been run by the maintainers yet: the torch path was verified
  end to end for Qwen3-ASR and Qwen3-TTS (MPS) and CosyVoice3 (CPU) on Apple
  Silicon. Reports welcome.
- Nemotron auto-detects language poorly for Chinese (20% CER); it wants an
  explicit `zh-CN`.

## [2.0.12] - 2026-05-17

### Added — TEN-VAD backend + `/vad` endpoint

The project gained an **engine-agnostic Voice Activity Detection** path
based on [TEN-VAD](https://huggingface.co/TEN-framework/ten-vad). TEN-VAD
is 731 KB, ships a native macOS arm64 binary, runs at ~0.016 RTF on M1,
and detects speech↔silence transitions ~100–300 ms faster than Silero
(which faster-whisper's internal `vad_filter` continues to use — this
addition is **additive**, not a replacement).

New module:
  - `localkin_service_audio/core/audio_processing/vad.py` with
    `TenVADBackend`, `SpeechSegment`, `detect_speech()` and the
    `SUPPORTED_VAD_BACKENDS` constant.

New HTTP endpoint (always available, regardless of which model the
server is started with):

```
POST /vad
  ?backend=ten-vad                  # currently only backend
  &threshold=0.5                    # 0.0–1.0 speech probability cutoff
  &min_speech_duration_ms=200       # drop bursts shorter than this
  &min_silence_duration_ms=200      # merge runs separated by less
  &speech_pad_ms=100                # pad each kept segment

Returns:
  {
    "backend": "ten-vad",
    "duration": 132.4,
    "speech_segments": [
      {"start": 1.2, "end": 5.8, "duration": 4.6},
      ...
    ],
    "total_speech_duration": 48.3
  }
```

New optional dependency: `pip install 'localkin-service-audio[vad]'`
(or just `pip install ten-vad`).

Tests: 12 new unit + integration tests in
`tests/unit/test_vad.py` cover the backend, the top-level helper, the
HTTP endpoint, and parameter pass-through. All 187 tests pass.

### Common workflow: chunk long audio before transcription

```bash
# 1) Find speech segments
SEGMENTS=$(curl -sX POST 'http://localhost:8000/vad' \
  -F 'file=@meeting.wav' | jq '.speech_segments')

# 2) Transcribe each (faster, skips silence, frees VRAM between chunks)
echo "$SEGMENTS" | jq -c '.[]' | while read -r seg; do
  start=$(echo "$seg" | jq .start)
  end=$(echo "$seg" | jq .end)
  ffmpeg -ss "$start" -to "$end" -i meeting.wav chunk.wav -y
  curl -sX POST 'http://localhost:8000/transcribe' -F 'file=@chunk.wav'
done
```

## [2.0.11] - 2026-04-26

### Added — Transcription Controls (resolves #2)

The `/transcribe` and `/v1/audio/transcriptions` endpoints now accept four
new query parameters that give callers explicit control over segmentation,
output shape, and VRAM usage:

- **`enable_vad`** (`bool`, default `True`) — Apply Voice Activity
  Detection to skip silence between speech. Currently honored by
  `faster-whisper`; ignored by engines that don't expose VAD.
- **`timestamps`** (`bool`, default `False`) — Include segment-level
  start/end times in JSON responses. Default `False` preserves the
  v2.0.x response shape for existing callers.
- **`response_format`** (`json` | `text` | `markdown` | `srt` | `vtt`,
  default `json`) — Output format. `markdown` produces a structured
  document with metadata and timestamped segment list. `srt` and `vtt`
  emit standard subtitle formats.
- **`chunk_length_s`** (`int`, optional) — Override the chunk length
  used for long audio. Honored by `faster-whisper` and the HuggingFace
  pipeline; ignored elsewhere. Useful for VRAM tuning.

Example:

```bash
# Markdown transcript with timestamps
curl -X POST 'http://localhost:8000/transcribe?response_format=markdown' \
  -F 'file=@meeting.wav'

# SRT subtitles
curl -X POST 'http://localhost:8000/transcribe?response_format=srt' \
  -F 'file=@video.wav' > captions.srt

# Tighter chunks for low-VRAM GPUs
curl -X POST 'http://localhost:8000/transcribe?chunk_length_s=15&enable_vad=true' \
  -F 'file=@long_audio.wav'
```

### Fixed
- **`faster-whisper` engine wired into the API**: Models with
  `engine="faster-whisper"` (e.g. `faster-whisper:base`, `:large-v3`,
  `:turbo`) were registered but the API server fell through to the
  HuggingFace pipeline, causing failures. Added a dedicated load path
  in `load_whisper_model()` that uses `FasterWhisperStrategy`.

### Internal
- New module `localkin_service_audio/api/transcription_format.py` with
  pure-Python formatters (`to_markdown`, `to_srt`, `to_vtt`) — engine
  agnostic, fully unit tested.
- New helpers in `server.py`: `_run_stt()` dispatches transcription to
  the right engine and returns a unified `(text, language, segments,
  duration)` tuple; `_build_transcription_response()` formats the HTTP
  response. Removes per-engine inline branching from the route handler.
- 22 new unit tests in `tests/unit/test_transcription_format.py`.

## [2.0.10] - 2026-03-10

### Fixed
- **Build failure**: Moved `openai-whisper` to optional dependency to resolve #1 build failure on systems without Rust/ffmpeg
- **Version mismatch**: Synced `__init__.py` version with `pyproject.toml` (was stuck at 2.0.9)
- **Bare `except:` clauses**: Replaced all bare `except:` with `except Exception:` across 11 files to avoid catching `KeyboardInterrupt`/`SystemExit`
- **HeartMuLa hardcoded path**: Removed fallback to `~/.cache/heartmula` — now consistently uses `LOCALKIN_HOME`
- **Pytest warnings**: Registered `@pytest.mark.slow` custom marker in `pyproject.toml`
- **Security**: Bumped `transformers>=4.53.0` (fixes 5 ReDoS CVEs) and `python-multipart>=0.0.22` (fixes arbitrary file write CVE)

### Removed
- **Legacy v1.x code** (~2,500 lines deleted):
  - `cli/cli_legacy.py` — old argparse CLI replaced by Click CLI in v2.0
  - `core/config_legacy.py` — replaced by `ModelRegistry` singleton
  - `core/audio_processing/stt_legacy.py` — replaced by STT strategy pattern
  - `core/audio_processing/tts_legacy.py` — replaced by TTS strategy pattern
- Legacy re-exports from `core/__init__.py` and `core/audio_processing/__init__.py`
- Legacy CLI fallback in `cli/main.py`

### Added
- **12 new models** in model registry (29 → 40 total):
  - **STT**: `whisper:large-v3-turbo`, `faster-whisper:large-v3-turbo` (6x faster than large-v3, MIT)
  - **STT**: `parakeet:0.6b` (NVIDIA, 10x faster than Whisper turbo, 25 languages)
  - **STT**: `canary:1b-v2` (NVIDIA, 25 languages, transcription + translation)
  - **STT**: `canary-qwen:2.5b` (NVIDIA, #1 HuggingFace ASR leaderboard, speech understanding)
  - **TTS**: `cosyvoice2:0.5b` (Alibaba, 30-50% fewer errors than v1, 9 langs + 18 Chinese dialects)
  - **TTS**: `qwen3-tts:0.6b`, `qwen3-tts:1.7b` (Alibaba, 97ms latency, voice design, 10 languages)
  - **TTS**: `orpheus:3b`, `orpheus:1b`, `orpheus:150m` (best emotional TTS, Llama-based, GGUF)
  - **TTS**: `dia:1.6b` (multi-speaker dialogue in one pass, laughter/coughing)

### Changed
- **Web UI (`ui/routes.py`)**: Migrated from legacy `get_models()`/`find_model()` to `model_registry` API; transcription uses `AudioEngine` instead of legacy `transcribe_audio()`
- **API server (`api/server.py`)**: Uses `model_registry` instead of legacy `find_model()`
- **Add-model command**: Self-contained models.json read/write, no longer depends on `config_legacy`
- **Package exports**: Top-level `__init__.py` now exports v2.0 API (`AudioEngine`, `model_registry`) instead of legacy functions

---

## [2.0.9] - 2026-02-15

### Fixed
- **Multilingual TTS missing deps**: Added `pyopenjtalk` (Japanese), `espeakng-loader` (French, Spanish, Italian, Hindi, Portuguese), `cn2an`, `jieba`, `pypinyin`, `pypinyin-dict` (Chinese), and `ordered-set` to dependencies — required by misaki (kokoro's text processor) but not declared in its own deps

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
