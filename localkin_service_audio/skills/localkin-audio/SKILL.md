---
name: localkin-audio
description: Local speech-to-text, text-to-speech, voice cloning and music with the `kin` CLI (LocalKin Audio). No API keys; everything runs on this machine. Use it to transcribe audio or video, read text aloud or make a voiceover, clone a voice from a recording, design a voice from a description, generate music, or run a local STT/TTS server. Strong on Chinese, Cantonese and mixed Chinese-English.
---

# LocalKin Audio

The `kin` command does speech-to-text, text-to-speech, voice cloning and music locally. A model
downloads the first time it's used. Fetch big ones ahead with `kin audio pull MODEL`.

## Pick a model

| Task | Model |
|---|---|
| Transcribe Chinese, Cantonese or mixed zh/en | `qwen3-asr:1.7b` (`qwen3-asr:0.6b` under 16 GB RAM) |
| Transcribe English or other languages fast on a CPU | `faster-whisper:large-v3-turbo` |
| Speak Chinese and English, mixed in one sentence | `qwen3-tts:0.6b`: 9 preset voices, tone with `--instruct` |
| Small, fast English speech (the default) | `kokoro` |
| Clone a voice from a 3-10 s recording | `qwen3-tts:1.7b-base` |
| Design a voice from a description | `qwen3-tts:1.7b-voicedesign` |
| Music: short instrumental, quick | `musicgen:small` |
| Music: a full song with lyrics (Apple Silicon) | `minimax-music3` |

On Apple Silicon these run on the GPU when the package has the `[mlx]` extra.
Elsewhere the `qwen3-*` models build their own environment on first use, once and in a few
minutes. That needs `uv` installed.

## Transcribe

```bash
kin audio transcribe recording.m4a -m qwen3-asr:1.7b
kin audio transcribe talk.mp3 -m qwen3-asr:1.7b -f json --timestamps
kin audio transcribe video.mp4 -m qwen3-asr:1.7b -f srt -o subtitles.srt
```

- Use `-l zh` (or `en`, ...) to fix the language. Without it the model detects the language.
- stdout carries only the transcript. Progress goes to stderr, so `-f json` can be parsed.
- m4a, mp4 and webm need ffmpeg; wav, flac, mp3 and ogg don't.

## Speak

Always pass `-o FILE --no-play`. Without `-o` the audio is played instead of saved.

```bash
kin audio tts "你好，欢迎回来。" -m qwen3-tts:0.6b -o hello.wav --no-play
kin audio tts "今天真不错！" -m qwen3-tts:0.6b --instruct "用开心的语气" -o happy.wav --no-play
kin audio tts "Hello there." -m qwen3-tts:0.6b --voice ryan --speed 1.1 -o hi.wav --no-play
kin audio tts -m qwen3-tts:0.6b --list-voices
```

## Clone or design a voice

```bash
kin audio tts "这句话用我的声音说。" -m qwen3-tts:1.7b-base --ref-audio me.wav --ref-text "录音里说的话" -o mine.wav --no-play
kin audio tts "欢迎收听本期节目。" -m qwen3-tts:1.7b-voicedesign --instruct "沉稳的中年男声，播音腔" -o host.wav --no-play
```

`--ref-text` is optional; giving the recording's transcript makes the clone closer.

## Music

```bash
kin audio music generate "calm lofi piano" -m musicgen:small -d 10 -o lofi.wav --no-play
kin audio music generate "欢快的流行，女声" -m minimax-music3 -l @lyrics.txt -o song.wav --no-play
```

Lyrics use `[verse]` / `[chorus]` tags. A full song takes minutes.

## Many requests: run a server

Every `kin audio` command loads its model again. For repeated calls, or for an app, keep the
models loaded:

```bash
kin audio serve --stt qwen3-asr:1.7b --tts qwen3-tts:0.6b --port 8000 &
curl -F file=@question.wav http://127.0.0.1:8000/v1/audio/transcriptions
curl http://127.0.0.1:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "你好", "instructions": "用开心的语气", "response_format": "mp3"}' -o answer.mp3
```

- The API is OpenAI-compatible, with `/synthesize` (`instruct`, `reference_audio`) and `/voices` alongside.
- Speech can stream: add `"stream_format": "audio"`; `pcm` streams by default.
- To reach it from other machines, add `--host 0.0.0.0 --api-key KEY`. Clients then send `Authorization: Bearer KEY`.
- Stop the server when you're done.
- MCP clients can use `kin mcp` instead, which offers the same tools. `kin mcp --http` serves them to remote agents.

## When something fails

- `kin audio models` lists every model; `kin audio models -t tts -l zh` filters them.
- A failed load prints the reason (a missing `uv`, the `[mlx]` extra not installed, ...) and exits
  non-zero.
- On macOS 27, Python 3.10 can't load scipy; use Python 3.11 or newer.
