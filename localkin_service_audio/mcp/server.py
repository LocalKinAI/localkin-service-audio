"""
MCP server: speech-to-text, text-to-speech, voice cloning and music for any
MCP client (Claude Code, Claude Desktop, Cursor, Codex, ...).

    claude mcp add localkin-audio -- kin mcp           # this machine, over stdio
    kin mcp --http --host 0.0.0.0 --api-key SECRET     # other machines, over HTTP

Over stdio the client starts the server and the tools read and write files
on this machine. Over HTTP (MCP's streamable HTTP, at /mcp) the server is
remote to its clients, so audio comes in as a URL or base64 and results go
back as a download URL (/files/...) and, on request, base64; paths can only
name files the server produced itself.

Each model loads on its first call (downloading it the first time) and stays
loaded; speech, cloning and transcription keep separate models so they don't
evict each other. Over stdio, stdout is reserved for the protocol: anything
else written to it, from Python or native code, goes to stderr. Works with
the MCP Python SDK 1.x and 2.x.
"""
import asyncio
import base64
import contextlib
import inspect
import io
import json
import os
import sys
import tempfile
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

MAX_DOWNLOAD = 200 * 1024 * 1024


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


def _audio_params(path_key: str, url_key: str, b64_key: str, what: str, remote: bool) -> Dict[str, Any]:
    where = ("a file this server produced (output_path of an earlier result)" if remote
             else "a file on this machine (wav, mp3, m4a, flac, ...)")
    return {
        path_key: {"type": "string", "description": f"{what}: {where}"},
        url_key: {"type": "string", "description": f"{what} as an http(s) URL to download"},
        b64_key: {"type": "string", "description": f"{what} as base64, or a data: URI"},
    }


def _output_params(remote: bool) -> Dict[str, Any]:
    params = {"return_base64": {"type": "boolean",
                                "description": "Also return the audio itself as base64 (audio_base64)"}}
    if not remote:
        params["output_path"] = {"type": "string",
                                 "description": "Where to write the audio (.wav, .flac, .ogg); a new file under "
                                                "~/.localkin-service-audio/outputs if omitted"}
    return params


def _tool_specs(d: Defaults, remote: bool = False) -> List[Dict[str, Any]]:
    result = ("Returns url, a download link for the file" if remote
              else "Returns output_path, the file written")
    return [
        {"name": "transcribe_audio",
         "description": "Transcribe a speech recording to text. Give the audio as audio_path, audio_url or "
                        "audio_base64. The default model handles Chinese (with Cantonese and dialects), "
                        "English and about 30 more languages, and detects the language itself. The first "
                        "call downloads the model.",
         "schema": {"type": "object", "properties": {
             **_audio_params("audio_path", "audio_url", "audio_base64", "The recording", remote),
             "language": {"type": "string", "description": "ISO code such as zh or en; omit to detect"},
             "model": {"type": "string", "description": f"Speech-to-text model (default {d.stt}); see list_models"},
         }}},
        {"name": "synthesize_speech",
         "description": "Read text aloud into an audio file. " + result + ". The default model mixes Chinese "
                        "and English in one sentence. `instruct` sets the tone (\"用开心的语气\", \"speak "
                        "slowly and softly\"); with model qwen3-tts:1.7b-voicedesign it describes the voice to "
                        "create instead. list_voices gives the voice ids.",
         "schema": {"type": "object", "required": ["text"], "properties": {
             "text": {"type": "string", "description": "What to say"},
             "voice": {"type": "string", "description": "Voice id from list_voices; omit for the default"},
             "instruct": {"type": "string", "description": "Tone, or a voice description for voice-design models"},
             "language": {"type": "string", "description": "ISO code; detected from the text if omitted"},
             "speed": {"type": "number", "description": "0.5-2.0, default 1.0"},
             "model": {"type": "string", "description": f"Text-to-speech model (default {d.tts})"},
             **_output_params(remote),
         }}},
        {"name": "clone_voice",
         "description": "Say text in the voice of a short recording: 3-10 s of one person speaking, given as "
                        "reference_audio, reference_url or reference_base64. " + result + ". Giving "
                        "reference_text (what the recording says) makes the voice closer.",
         "schema": {"type": "object", "required": ["text"], "properties": {
             **_audio_params("reference_audio", "reference_url", "reference_base64",
                             "Recording of the voice to clone", remote),
             "text": {"type": "string", "description": "What the cloned voice should say"},
             "reference_text": {"type": "string", "description": "Transcript of the recording"},
             "language": {"type": "string", "description": "ISO code; detected from the text if omitted"},
             "model": {"type": "string", "description": f"Voice-cloning model (default {d.clone})"},
             **_output_params(remote),
         }}},
        {"name": "generate_music",
         "description": "Compose music from a description. " + result + ". The default musicgen:small makes "
                        "short instrumentals quickly. For full songs with lyrics use minimax-music3 (Apple "
                        "Silicon via MLX, or a ComfyUI that has it); heartmula:3b takes Chinese prompts. "
                        "Songs take minutes.",
         "schema": {"type": "object", "required": ["prompt"], "properties": {
             "prompt": {"type": "string", "description": "Style, mood, instruments, e.g. \"calm lofi piano\""},
             "lyrics": {"type": "string", "description": "Lyrics with [verse]/[chorus] tags (song models)"},
             "duration": {"type": "integer", "description": "Seconds; each model has its own default"},
             "model": {"type": "string", "description": f"Music model (default {d.music}); see list_models"},
             **_output_params(remote),
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
         "description": "Duration, sample rate, channels and format of a recording given as audio_path, "
                        "audio_url or audio_base64.",
         "schema": {"type": "object", "properties": _audio_params("audio_path", "audio_url", "audio_base64",
                                                                 "The recording", remote)}},
    ]


def outputs_dir() -> Path:
    from ..core.config.settings import _default_home
    return _default_home() / "outputs"


def _existing(path: str) -> str:
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise FileNotFoundError(f"No such file: {path}")
    return full


def _sniff_suffix(head: bytes) -> str:
    """A file extension for audio bytes, so decoders that go by name find their way."""
    if head[:4] == b"RIFF":
        return ".wav"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if head[4:8] == b"ftyp":
        return ".m4a"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".bin"


def _temp_audio(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=_sniff_suffix(data[:12]))
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _download(url: str) -> str:
    """An http(s) URL fetched to a temporary file, up to 200 MB."""
    import urllib.request

    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Audio URLs must be http(s)")
    request = urllib.request.Request(url, headers={"User-Agent": "localkin-service-audio"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("The audio at that URL is larger than 200 MB")
    return _temp_audio(data)


def _decode_base64(value: str) -> str:
    import binascii

    if value.startswith("data:"):
        value = value.partition(",")[2]
    try:
        data = base64.b64decode(value)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"That isn't valid base64 audio: {e}")
    return _temp_audio(data)


def _output_path(path: Optional[str], prefix: str) -> str:
    if path:
        full = Path(path).expanduser().resolve()
    else:
        full = outputs_dir() / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.wav"
    full.parent.mkdir(parents=True, exist_ok=True)
    return str(full)


def _segment(s) -> Dict[str, Any]:
    get = s.get if isinstance(s, dict) else (lambda k, d=None: getattr(s, k, d))
    return {"start": get("start"), "end": get("end"), "text": get("text")}


class AudioTools:
    """The tools themselves, independent of the MCP SDK.

    ``remote``: the clients are on other machines, so audio arrives as a URL
    or base64, results are returned as ``public_url``/files/<name>, and paths
    may only name files in the outputs directory.
    """

    def __init__(self, defaults: Defaults, remote: bool = False, public_url: Optional[str] = None):
        from ..core.audio_processing.engine import AudioEngine

        self.defaults = defaults
        self.remote = remote
        self.public_url = (public_url or "").rstrip("/")
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

    def _fetch_audio(self, args, path_key: str, url_key: str, b64_key: str, convert: bool):
        """(local file, temporary files to remove) for audio given as a path,
        a URL or base64; a WAV if ``convert``."""
        from ..core.audio_processing.shared import readable_wav

        given = [k for k in (path_key, url_key, b64_key) if args.get(k)]
        if len(given) != 1:
            raise ValueError(f"Give the audio as one of {path_key}, {url_key} or {b64_key}")
        temporary = []
        own = f"{self.public_url}/files/" if self.public_url else None
        if args.get(url_key) and own and args[url_key].startswith(own):
            # One of this server's own results: read it, don't fetch it.
            path = _existing(str(outputs_dir() / os.path.basename(args[url_key][len(own):])))
        elif args.get(url_key):
            path = _download(args[url_key])
            temporary.append(path)
        elif args.get(b64_key):
            path = _decode_base64(args[b64_key])
            temporary.append(path)
        else:
            path = _existing(args[path_key])
            if self.remote and not Path(path).resolve().is_relative_to(outputs_dir().resolve()):
                raise PermissionError(f"This server is remote: give {url_key} or {b64_key} "
                                      "(paths only reach files it produced)")
        if convert:
            try:
                path, converted = readable_wav(path)
            except Exception:
                for t in temporary:
                    os.unlink(t)
                raise
            if converted:
                temporary.append(path)
        return path, temporary

    @contextlib.asynccontextmanager
    async def _audio_in(self, args, path_key: str, url_key: str, b64_key: str, convert: bool = True):
        """The audio a call gives, as a local file; fetched and converted
        copies are removed afterwards. The fetching runs off the event loop."""
        path, temporary = await asyncio.to_thread(self._fetch_audio, args, path_key, url_key, b64_key, convert)
        try:
            yield path
        finally:
            for t in temporary:
                os.unlink(t)

    def _audio_out(self, result, args, prefix: str) -> Dict[str, Any]:
        """Save ``result`` and describe it for the client."""
        if self.remote and args.get("output_path"):
            raise PermissionError("This server is remote: results stay on it and come back as a url")
        path = result.save(_output_path(args.get("output_path"), prefix))
        duration = result.duration or len(result.audio) / result.sample_rate
        out = {"output_path": path, "duration": round(duration, 2), "sample_rate": result.sample_rate}
        if self.remote:
            out["url"] = f"{self.public_url}/files/{os.path.basename(path)}"
        if args.get("return_base64"):
            out["audio_base64"] = base64.b64encode(Path(path).read_bytes()).decode()
            out["mime_type"] = "audio/wav"
        return out

    async def transcribe_audio(self, args):
        model = args.get("model") or self.defaults.stt
        language = None if args.get("language") in (None, "", "auto") else args["language"]
        async with self._audio_in(args, "audio_path", "audio_url", "audio_base64") as audio:
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
            out = await asyncio.to_thread(self._audio_out, r, args, "speech")
        return {**out, "model": model, "voice": r.voice}

    async def clone_voice(self, args):
        model = args.get("model") or self.defaults.clone
        extra = {"language": args["language"]} if args.get("language") else {}
        async with self._audio_in(args, "reference_audio", "reference_url", "reference_base64") as reference:
            async with self._locks["clone"]:
                engine = await asyncio.to_thread(self._load, "clone", model)
                config = getattr(engine._tts_strategy, "model_config", None)
                if config is not None and not config.supports_voice_cloning:
                    raise ValueError(f"{model} doesn't clone voices; try {self.defaults.clone}")
                r = await asyncio.to_thread(engine.clone_voice, reference, args["text"],
                                            args.get("reference_text"), **extra)
                out = await asyncio.to_thread(self._audio_out, r, args, "clone")
        return {**out, "model": model}

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
            out = await asyncio.to_thread(self._audio_out, r, args, "music")
        return {**out, "model": model}

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

        async with self._audio_in(args, "audio_path", "audio_url", "audio_base64", convert=False) as audio:
            info = sf.info(audio)
        out = {"duration": info.duration, "sample_rate": info.samplerate, "channels": info.channels,
               "format": info.format, "subtype": info.subtype}
        if args.get("audio_path"):
            out["path"] = _existing(args["audio_path"])
        return out


def build_server(audio: AudioTools) -> "Server":
    """The low-level MCP server for ``audio``, on either SDK generation."""
    from .. import __version__

    tools = [types.Tool(name=s["name"], description=s["description"], inputSchema=s["schema"])
             for s in _tool_specs(audio.defaults, audio.remote)]
    if audio.remote:
        instructions = ("Local speech-to-text, text-to-speech, voice cloning and music, running on another "
                        "machine: pass audio as a URL or base64; results come back as a download url "
                        "(and as base64 when return_base64 is true).")
    else:
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


def default_public_url(host: str, port: int) -> str:
    """How clients reach a server bound to ``host``: its LAN address when it
    listens on every interface."""
    if host in ("0.0.0.0", "::", ""):
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))  # picks the outgoing interface; sends nothing
                host = s.getsockname()[0]
        except OSError:
            host = socket.gethostname()
    return f"http://{host}:{port}"


class _MCPEndpoint:
    """The streamable-HTTP session manager as a plain ASGI app."""

    def __init__(self, manager):
        self.manager = manager

    async def __call__(self, scope, receive, send):
        await self.manager.handle_request(scope, receive, send)


def create_http_app(audio: AudioTools, api_key: Optional[str] = None):
    """The remote MCP server: MCP at /mcp, results at /files/<name>, /health."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route

    from ..api.auth import BearerAuth

    manager = StreamableHTTPSessionManager(app=build_server(audio))

    async def files(request):
        path = outputs_dir() / os.path.basename(request.path_params["name"])
        if not path.is_file():
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return FileResponse(path)

    async def health(request):
        return JSONResponse({"status": "healthy", "transport": "streamable-http",
                             "defaults": asdict(audio.defaults)})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    middleware = [Middleware(BearerAuth, api_key=api_key, open_paths={"/health"})] if api_key else []
    return Starlette(routes=[Route("/mcp", endpoint=_MCPEndpoint(manager), methods=["GET", "POST", "DELETE"]),
                             Route("/files/{name}", files), Route("/health", health)],
                     middleware=middleware, lifespan=lifespan)


def run_http_server(host: str = "127.0.0.1", port: int = 8765, api_key: Optional[str] = None,
                    public_url: Optional[str] = None, stt: Optional[str] = None, tts: Optional[str] = None,
                    clone: Optional[str] = None, music: Optional[str] = None) -> None:
    """Serve MCP over streamable HTTP for clients on other machines."""
    if not MCP_AVAILABLE:
        print(INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)
    import uvicorn

    public_url = public_url or default_public_url(host, port)
    audio = AudioTools(default_models(stt, tts, clone, music), remote=True, public_url=public_url)
    print(f"MCP over HTTP at {public_url}/mcp" + ("" if api_key else "  (no API key: anyone who can reach it can use it)"),
          file=sys.stderr)
    uvicorn.run(create_http_app(audio, api_key), host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
