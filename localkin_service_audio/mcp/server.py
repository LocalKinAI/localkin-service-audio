"""
MCP server: speech-to-text, text-to-speech, voice cloning and music for any
MCP client (Claude Code, Claude Desktop, Cursor, Codex, ...).

    claude mcp add localkin-audio -- kin mcp
    kin mcp --stt qwen3-asr:0.6b --tts kokoro      # other default models

The client starts the server over stdio. Each model loads on its first call
(downloading it the first time) and stays loaded for the session; speech,
cloning and transcription keep separate models so they don't evict each
other. Model libraries print freely, so stdout is reserved for the protocol:
anything else written to it, from Python or native code, goes to stderr.
Works with the MCP Python SDK 1.x and 2.x.
"""
import asyncio
import contextlib
import inspect
import io
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

INSTALL_HINT = 'The MCP server needs the mcp package: pip install "localkin-service-audio[mcp]"'


@dataclass
class Defaults:
    """The model each kind of call uses when it doesn't name one."""
    stt: str
    tts: str
    clone: str
    music: str


def default_models(stt: Optional[str] = None, tts: Optional[str] = None,
                   clone: Optional[str] = None, music: Optional[str] = None) -> Defaults:
    """The given model, else LOCALKIN_{STT,TTS,CLONE,MUSIC}_MODEL, else one
    sized to this machine. Qwen3 runs on MLX on Apple Silicon (with the
    [mlx] extra) and in an auto-built torch environment elsewhere."""
    from ..cli.utils.device import get_ram_gb

    big = get_ram_gb() >= 16
    env = os.environ.get
    return Defaults(
        stt=stt or env("LOCALKIN_STT_MODEL") or ("qwen3-asr:1.7b" if big else "qwen3-asr:0.6b"),
        tts=tts or env("LOCALKIN_TTS_MODEL") or "qwen3-tts:0.6b",
        clone=clone or env("LOCALKIN_CLONE_MODEL") or ("qwen3-tts:1.7b-base" if big else "qwen3-tts:0.6b-base"),
        music=music or env("LOCALKIN_MUSIC_MODEL") or "musicgen:small",
    )


def _tool_specs(d: Defaults) -> List[Dict[str, Any]]:
    path = {"type": "string"}
    return [
        {"name": "transcribe_audio",
         "description": "Transcribe a speech recording to text. The default model handles Chinese (with "
                        "Cantonese and dialects), English and about 30 more languages, and detects the "
                        "language itself. The first call downloads the model.",
         "schema": {"type": "object", "required": ["audio_path"], "properties": {
             "audio_path": {**path, "description": "Audio file on this machine (wav, mp3, m4a, flac, ...)"},
             "language": {"type": "string", "description": "ISO code such as zh or en; omit to detect"},
             "model": {"type": "string", "description": f"Speech-to-text model (default {d.stt}); see list_models"},
         }}},
        {"name": "synthesize_speech",
         "description": "Read text aloud into an audio file and return its path. The default model mixes "
                        "Chinese and English in one sentence. `instruct` sets the tone (\"用开心的语气\", "
                        "\"speak slowly and softly\"); with model qwen3-tts:1.7b-voicedesign it describes "
                        "the voice to create instead. list_voices gives the voice ids.",
         "schema": {"type": "object", "required": ["text"], "properties": {
             "text": {"type": "string", "description": "What to say"},
             "voice": {"type": "string", "description": "Voice id from list_voices; omit for the default"},
             "instruct": {"type": "string", "description": "Tone, or a voice description for voice-design models"},
             "language": {"type": "string", "description": "ISO code; detected from the text if omitted"},
             "speed": {"type": "number", "description": "0.5-2.0, default 1.0"},
             "model": {"type": "string", "description": f"Text-to-speech model (default {d.tts})"},
             "output_path": {**path, "description": "Where to write the audio (.wav, .flac, .ogg); a new "
                                                    "file under ~/.localkin-service-audio/outputs if omitted"},
         }}},
        {"name": "clone_voice",
         "description": "Say text in the voice of a short recording: 3-10 s of one person speaking. "
                        "Giving reference_text (what the recording says) makes the voice closer.",
         "schema": {"type": "object", "required": ["reference_audio", "text"], "properties": {
             "reference_audio": {**path, "description": "Recording of the voice to clone"},
             "text": {"type": "string", "description": "What the cloned voice should say"},
             "reference_text": {"type": "string", "description": "Transcript of the recording"},
             "language": {"type": "string", "description": "ISO code; detected from the text if omitted"},
             "model": {"type": "string", "description": f"Voice-cloning model (default {d.clone})"},
             "output_path": {**path, "description": "Where to write the audio; generated if omitted"},
         }}},
        {"name": "generate_music",
         "description": "Compose music from a description and return the file's path. The default "
                        "musicgen:small makes short instrumentals quickly. For full songs with lyrics use "
                        "minimax-music3 (Apple Silicon via MLX, or a ComfyUI that has it); heartmula:3b "
                        "takes Chinese prompts. Songs take minutes.",
         "schema": {"type": "object", "required": ["prompt"], "properties": {
             "prompt": {"type": "string", "description": "Style, mood, instruments, e.g. \"calm lofi piano\""},
             "lyrics": {"type": "string", "description": "Lyrics with [verse]/[chorus] tags (song models)"},
             "duration": {"type": "integer", "description": "Seconds; each model has its own default"},
             "model": {"type": "string", "description": f"Music model (default {d.music}); see list_models"},
             "output_path": {**path, "description": "Where to write the audio; generated if omitted"},
         }}},
        {"name": "list_models",
         "description": "List the speech-to-text, text-to-speech and music models, and the defaults in use.",
         "schema": {"type": "object", "properties": {
             "type": {"type": "string", "enum": ["all", "stt", "tts", "music"], "description": "Default all"},
             "language": {"type": "string", "description": "Only models that support this ISO code"},
         }}},
        {"name": "list_voices",
         "description": "List the preset voices of a text-to-speech model (the default one unless given).",
         "schema": {"type": "object", "properties": {
             "model": {"type": "string", "description": f"Text-to-speech model (default {d.tts})"},
         }}},
        {"name": "get_audio_info",
         "description": "Duration, sample rate, channels and format of an audio file.",
         "schema": {"type": "object", "required": ["audio_path"], "properties": {"audio_path": path}}},
    ]


def _existing(path: str) -> str:
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise FileNotFoundError(f"No such file: {path}")
    return full


@contextlib.contextmanager
def _as_wav(path: str):
    """``path`` as a WAV every engine opens (mp3, m4a, ... are converted to
    a temporary file, removed afterwards)."""
    from ..core.audio_processing.shared import readable_wav

    wav, temporary = readable_wav(_existing(path))
    try:
        yield wav
    finally:
        if temporary:
            os.unlink(wav)


def _output_path(path: Optional[str], prefix: str) -> str:
    if path:
        full = Path(path).expanduser().resolve()
    else:
        from ..core.config.settings import _default_home
        full = _default_home() / "outputs" / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.wav"
    full.parent.mkdir(parents=True, exist_ok=True)
    return str(full)


def _segment(s) -> Dict[str, Any]:
    get = s.get if isinstance(s, dict) else (lambda k, d=None: getattr(s, k, d))
    return {"start": get("start"), "end": get("end"), "text": get("text")}


class AudioTools:
    """The tools themselves, independent of the MCP SDK."""

    def __init__(self, defaults: Defaults):
        from ..core.audio_processing.engine import AudioEngine

        self.defaults = defaults
        # One engine per role, so a clone between two speeches doesn't
        # reload either model.
        self._engines = {"stt": AudioEngine(), "tts": AudioEngine(), "clone": AudioEngine()}
        self._locks = {role: asyncio.Lock() for role in ("stt", "tts", "clone", "music")}
        self._music: Dict[str, Any] = {}

    async def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handler = {
            "transcribe_audio": self.transcribe_audio, "synthesize_speech": self.synthesize_speech,
            "clone_voice": self.clone_voice, "generate_music": self.generate_music,
            "list_models": self.list_models, "list_voices": self.list_voices,
            "get_audio_info": self.get_audio_info,
        }.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        return await handler(args or {})

    def _load(self, role: str, model: str):
        engine = self._engines[role]
        stt = role == "stt"
        if (engine._stt_model_name if stt else engine._tts_model_name) == model:
            return engine
        if not (engine.load_stt if stt else engine.load_tts)(model):
            strategy = engine._stt_strategy if stt else engine._tts_strategy
            reason = getattr(strategy, "load_error", None)
            raise RuntimeError(f"Couldn't load {model}" + (f": {reason}" if reason else ""))
        return engine

    async def transcribe_audio(self, args):
        model = args.get("model") or self.defaults.stt
        language = None if args.get("language") in (None, "", "auto") else args["language"]
        with _as_wav(args["audio_path"]) as audio:
            async with self._locks["stt"]:
                engine = await asyncio.to_thread(self._load, "stt", model)
                r = await asyncio.to_thread(engine.transcribe, audio, language=language)
        out = {"text": r.text, "language": r.language, "duration": r.duration, "model": model}
        if r.segments:
            out["segments"] = [_segment(s) for s in r.segments]
        if r.emotion:
            out["emotion"] = r.emotion
        return out

    async def synthesize_speech(self, args):
        model = args.get("model") or self.defaults.tts
        extra = {k: args[k] for k in ("language", "instruct") if args.get(k)}
        async with self._locks["tts"]:
            engine = await asyncio.to_thread(self._load, "tts", model)
            r = await asyncio.to_thread(engine.synthesize, args["text"], voice=args.get("voice"),
                                        speed=float(args.get("speed") or 1.0), **extra)
            path = await asyncio.to_thread(r.save, _output_path(args.get("output_path"), "speech"))
        return {"output_path": path, "duration": round(r.duration, 2), "sample_rate": r.sample_rate,
                "model": model, "voice": r.voice}

    async def clone_voice(self, args):
        _existing(args["reference_audio"])
        model = args.get("model") or self.defaults.clone
        extra = {"language": args["language"]} if args.get("language") else {}
        async with self._locks["clone"]:
            engine = await asyncio.to_thread(self._load, "clone", model)
            config = getattr(engine._tts_strategy, "model_config", None)
            if config is not None and not config.supports_voice_cloning:
                raise ValueError(f"{model} doesn't clone voices; try {self.defaults.clone}")
            with _as_wav(args["reference_audio"]) as reference:
                r = await asyncio.to_thread(engine.clone_voice, reference, args["text"],
                                            args.get("reference_text"), **extra)
            path = await asyncio.to_thread(r.save, _output_path(args.get("output_path"), "clone"))
        return {"output_path": path, "duration": round(r.duration, 2), "sample_rate": r.sample_rate,
                "model": model}

    async def generate_music(self, args):
        from ..music.generate import generate_music, load_music

        model = args.get("model") or self.defaults.music
        async with self._locks["music"]:
            engine = self._music.get(model)
            if engine is None:
                for old in self._music.values():  # music models are big: one at a time
                    old.unload()
                self._music.clear()
                engine = self._music[model] = await asyncio.to_thread(load_music, model)
            r = await asyncio.to_thread(generate_music, engine, args["prompt"], lyrics=args.get("lyrics"),
                                        duration=args.get("duration"))
            path = await asyncio.to_thread(r.save, _output_path(args.get("output_path"), "music"))
        return {"output_path": path, "duration": round(r.duration, 2), "sample_rate": r.sample_rate,
                "model": model}

    async def list_models(self, args):
        from ..core.config import model_registry
        from ..music.generate import music_models

        kind, language = args.get("type") or "all", args.get("language")
        rows = []
        if kind in ("all", "stt", "tts"):
            for m in model_registry.list_all():
                if kind != "all" and m.type.value != kind:
                    continue
                if language and language not in (m.languages or []):
                    continue
                rows.append({"name": m.name, "type": m.type.value, "languages": m.languages,
                             "description": m.description})
        if kind in ("all", "music") and not language:
            rows += [{"name": r["name"], "type": "music", "description": r["description"]}
                     for r in music_models()]
        return {"defaults": asdict(self.defaults), "models": rows, "total": len(rows)}

    async def list_voices(self, args):
        model = args.get("model") or self.defaults.tts
        async with self._locks["tts"]:
            engine = await asyncio.to_thread(self._load, "tts", model)
            voices = engine.list_voices()
        return {"model": model, "total": len(voices),
                "voices": [{"id": v.id, "name": v.name, "language": v.language, "gender": v.gender}
                           for v in voices]}

    async def get_audio_info(self, args):
        import soundfile as sf

        audio = _existing(args["audio_path"])
        info = sf.info(audio)
        return {"path": audio, "duration": info.duration, "sample_rate": info.samplerate,
                "channels": info.channels, "format": info.format, "subtype": info.subtype}


def build_server(audio: AudioTools) -> "Server":
    """The low-level MCP server for ``audio``, on either SDK generation."""
    from .. import __version__

    tools = [types.Tool(name=s["name"], description=s["description"], inputSchema=s["schema"])
             for s in _tool_specs(audio.defaults)]
    instructions = ("Local speech-to-text, text-to-speech, voice cloning and music. Paths are files on "
                    "the machine running this server; results are written there too.")

    async def call(name, arguments):
        result = await audio.call(name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    accepted = inspect.signature(Server.__init__).parameters
    if "on_call_tool" in accepted:  # SDK 2.x: handlers are passed in
        async def on_list_tools(ctx, params):
            return types.ListToolsResult(tools=tools)

        async def on_call_tool(ctx, params):
            try:
                return types.CallToolResult(content=await call(params.name, params.arguments))
            except Exception as e:
                return types.CallToolResult(isError=True, content=[
                    types.TextContent(type="text", text=f"{type(e).__name__}: {e}")])

        return Server("localkin-audio", version=__version__, instructions=instructions,
                      on_list_tools=on_list_tools, on_call_tool=on_call_tool)

    # SDK 1.x: decorators; an exception becomes an isError result.
    options = {k: v for k, v in (("version", __version__), ("instructions", instructions)) if k in accepted}
    server = Server("localkin-audio", **options)

    @server.list_tools()
    async def list_tools():
        return tools

    @server.call_tool()
    async def call_tool(name, arguments):
        return await call(name, arguments)

    return server


def create_server() -> "Server":
    """The MCP server with this machine's default models."""
    if not MCP_AVAILABLE:
        raise ImportError(INSTALL_HINT)
    return build_server(AudioTools(default_models()))


def _take_stdout() -> io.TextIOWrapper:
    """A private handle on stdout for the protocol. stdout itself, Python's
    and file descriptor 1, then points at stderr, so a library that prints
    can't corrupt the JSON-RPC stream."""
    sys.stdout.flush()
    fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return io.TextIOWrapper(os.fdopen(fd, "wb", buffering=0), encoding="utf-8",
                            line_buffering=True, write_through=True)


def run_server(stt: Optional[str] = None, tts: Optional[str] = None,
               clone: Optional[str] = None, music: Optional[str] = None) -> None:
    """Serve over stdio until the client disconnects."""
    if not MCP_AVAILABLE:
        print(INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)
    import anyio

    protocol = anyio.wrap_file(_take_stdout())
    server = build_server(AudioTools(default_models(stt, tts, clone, music)))

    async def main():
        async with stdio_server(stdout=protocol) as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    run_server()
