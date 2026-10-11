"""Tests for the MCP server (`kin mcp`).

The tools run against a fake AudioEngine, so nothing downloads. The last
test drives the real server through the MCP SDK's own client over stdio;
it runs where the `mcp` package is installed (the [mcp] extra).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from localkin_service_audio.core.types import AudioResult, ModelConfig, ModelType, Segment, TranscriptionResult, VoiceInfo
from localkin_service_audio.mcp import server as mcp_server


class FakeEngine:
    """Stands in for AudioEngine: records calls, loads instantly."""
    instances = []

    def __init__(self):
        self._stt_model_name = self._tts_model_name = None
        self._stt_strategy = self._tts_strategy = None
        self.loads, self.calls = [], []
        FakeEngine.instances.append(self)

    def load_stt(self, model):
        self.loads.append(model)
        self._stt_model_name = model
        return True

    def load_tts(self, model):
        self.loads.append(model)
        if model == "broken":
            self._tts_strategy = types.SimpleNamespace(load_error="needs uv")
            return False
        cloning = "base" in model
        self._tts_strategy = types.SimpleNamespace(model_config=ModelConfig(
            name=model, type=ModelType.TTS, engine="fake", features=["voice_cloning"] if cloning else []))
        self._tts_model_name = model
        return True

    def transcribe(self, audio, language=None):
        self.calls.append(("transcribe", audio, language))
        return TranscriptionResult(text="你好世界", language="zh", duration=1.0,
                                   segments=[Segment(text="你好世界", start=0.0, end=1.0)])

    def synthesize(self, text, voice=None, speed=1.0, **kwargs):
        self.calls.append(("synthesize", text, voice, speed, kwargs))
        return AudioResult(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000, voice=voice or "vivian",
                           duration=0.1)

    def clone_voice(self, reference, text, reference_text=None, **kwargs):
        self.calls.append(("clone", reference, text, reference_text, kwargs))
        return AudioResult(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000, duration=0.1)

    def list_voices(self):
        return [VoiceInfo(id="vivian", name="Vivian", language="zh", gender="female")]


@pytest.fixture
def tools(monkeypatch, tmp_path):
    from localkin_service_audio.core.audio_processing import engine

    FakeEngine.instances = []
    monkeypatch.setattr(engine, "AudioEngine", FakeEngine)
    monkeypatch.setenv("LOCALKIN_HOME", str(tmp_path / "home"))
    defaults = mcp_server.Defaults(stt="qwen3-asr:1.7b", tts="qwen3-tts:0.6b",
                                   clone="qwen3-tts:1.7b-base", music="musicgen:small")
    return mcp_server.AudioTools(defaults)


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "in.wav"
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)
    return str(path)


def run(coro):
    return asyncio.run(coro)


def test_default_models_prefer_argument_then_env_then_machine(monkeypatch):
    from localkin_service_audio.cli.utils import device

    for var in ("LOCALKIN_STT_MODEL", "LOCALKIN_TTS_MODEL", "LOCALKIN_CLONE_MODEL", "LOCALKIN_MUSIC_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(device, "get_ram_gb", lambda: 8.0)
    small = mcp_server.default_models()
    assert (small.stt, small.tts, small.clone, small.music) == \
        ("qwen3-asr:0.6b", "qwen3-tts:0.6b", "qwen3-tts:0.6b-base", "musicgen:small")
    monkeypatch.setattr(device, "get_ram_gb", lambda: 64.0)
    assert mcp_server.default_models().stt == "qwen3-asr:1.7b"
    monkeypatch.setenv("LOCALKIN_STT_MODEL", "sensevoice:small")
    assert mcp_server.default_models().stt == "sensevoice:small"
    assert mcp_server.default_models(stt="fun-asr:nano").stt == "fun-asr:nano"


def test_transcribe_returns_text_and_segments(tools, wav):
    out = run(tools.call("transcribe_audio", {"audio_path": wav, "language": "auto"}))
    assert out["text"] == "你好世界" and out["model"] == "qwen3-asr:1.7b"
    assert out["segments"] == [{"start": 0.0, "end": 1.0, "text": "你好世界"}]
    stt = FakeEngine.instances[0]
    assert stt.calls[-1] == ("transcribe", wav, None)  # "auto" means detect


def test_speech_passes_tone_language_speed_and_writes_the_file(tools, tmp_path):
    target = tmp_path / "out" / "hi.wav"
    out = run(tools.call("synthesize_speech", {"text": "你好", "instruct": "用开心的语气", "language": "zh",
                                               "speed": 1.2, "voice": "serena", "output_path": str(target)}))
    assert out["output_path"] == str(target) and target.exists()
    tts = FakeEngine.instances[1]
    assert tts.calls[-1] == ("synthesize", "你好", "serena", 1.2, {"language": "zh", "instruct": "用开心的语气"})


def test_default_output_goes_under_the_home_outputs_dir(tools, tmp_path):
    out = run(tools.call("synthesize_speech", {"text": "hello"}))
    assert out["output_path"].startswith(str(tmp_path / "home" / "outputs")) and os.path.exists(out["output_path"])


def test_speech_and_clone_keep_separate_models(tools, wav):
    run(tools.call("synthesize_speech", {"text": "a"}))
    run(tools.call("clone_voice", {"reference_audio": wav, "text": "b", "reference_text": "参考"}))
    run(tools.call("synthesize_speech", {"text": "c"}))
    _, tts, clone = FakeEngine.instances
    assert tts.loads == ["qwen3-tts:0.6b"] and clone.loads == ["qwen3-tts:1.7b-base"]
    assert clone.calls[-1][:4] == ("clone", wav, "b", "参考")


def test_clone_refuses_a_model_that_cannot_clone(tools, wav):
    with pytest.raises(ValueError, match="doesn't clone"):
        run(tools.call("clone_voice", {"reference_audio": wav, "text": "b", "model": "qwen3-tts:0.6b"}))


def test_load_failure_carries_the_reason(tools):
    with pytest.raises(RuntimeError, match="Couldn't load broken: needs uv"):
        run(tools.call("synthesize_speech", {"text": "x", "model": "broken"}))


def test_missing_file_and_unknown_tool_are_errors(tools):
    with pytest.raises(FileNotFoundError):
        run(tools.call("transcribe_audio", {"audio_path": "/nope.wav"}))
    with pytest.raises(ValueError, match="Unknown tool"):
        run(tools.call("dance", {}))


def test_music_loads_one_model_at_a_time(tools, monkeypatch):
    from localkin_service_audio.music import generate

    unloaded = []

    class FakeMusic:
        def __init__(self, model):
            self.model = model

        def unload(self):
            unloaded.append(self.model)

    monkeypatch.setattr(generate, "load_music", lambda model: FakeMusic(model))
    monkeypatch.setattr(generate, "generate_music", lambda engine, prompt, **kw: AudioResult(
        audio=np.zeros(3200, dtype=np.float32), sample_rate=32000, duration=0.1))
    run(tools.call("generate_music", {"prompt": "lofi"}))
    run(tools.call("generate_music", {"prompt": "song", "model": "minimax-music3", "lyrics": "[verse] hi"}))
    assert unloaded == ["musicgen:small"] and list(tools._music) == ["minimax-music3"]


def test_list_models_has_defaults_and_music(tools):
    out = run(tools.call("list_models", {"type": "music"}))
    assert out["defaults"]["tts"] == "qwen3-tts:0.6b"
    assert {"musicgen:small", "minimax-music3"} <= {m["name"] for m in out["models"]}
    zh = run(tools.call("list_models", {"type": "stt", "language": "zh"}))
    assert zh["total"] > 5 and all(m["type"] == "stt" for m in zh["models"])


def test_take_stdout_keeps_stdout_for_the_protocol():
    """Python prints and native writes to fd 1 land on stderr."""
    code = ("import os; from localkin_service_audio.mcp.server import _take_stdout; out = _take_stdout(); "
            "print('python noise'); os.write(1, b'native noise\\n'); out.write('PROTOCOL\\n')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "PROTOCOL\n"
    assert "python noise" in r.stderr and "native noise" in r.stderr


def test_real_mcp_client_round_trip(wav):
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    def error_flag(result):
        return getattr(result, "is_error", getattr(result, "isError", None))

    async def session_calls():
        params = StdioServerParameters(command=sys.executable, args=["-m", "localkin_service_audio.mcp.server"],
                                       env={**os.environ, "LOCALKIN_TTS_MODEL": "kokoro"})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                info = await session.call_tool("get_audio_info", {"audio_path": wav})
                missing = await session.call_tool("transcribe_audio", {"audio_path": "/nope.wav"})
                return names, info, missing

    names, info, missing = anyio.run(session_calls)
    assert names == ["transcribe_audio", "synthesize_speech", "clone_voice", "generate_music",
                     "list_models", "list_voices", "get_audio_info"]
    assert '"sample_rate": 16000' in info.content[0].text and not error_flag(info)
    assert error_flag(missing) and "No such file" in missing.content[0].text


def test_transcribe_converts_mp3_and_removes_the_copy(tools, tmp_path):
    mp3 = tmp_path / "voice.mp3"
    sf.write(mp3, np.full(16000, 0.05, dtype=np.float32), 16000, format="MP3")
    run(tools.call("transcribe_audio", {"audio_path": str(mp3)}))
    passed = FakeEngine.instances[0].calls[-1][1]
    assert passed.endswith(".wav") and passed != str(mp3) and not os.path.exists(passed)


def _b64_wav():
    import base64
    import io

    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("prefix", ["", "data:audio/wav;base64,"])
def test_transcribe_takes_base64_and_removes_the_copy(tools, prefix):
    out = run(tools.call("transcribe_audio", {"audio_base64": prefix + _b64_wav()}))
    passed = FakeEngine.instances[0].calls[-1][1]
    assert out["text"] == "你好世界" and passed.endswith(".wav") and not os.path.exists(passed)


def test_transcribe_downloads_a_url(tools, wav):
    import functools
    import http.server
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(wav))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/{os.path.basename(wav)}"
        assert run(tools.call("transcribe_audio", {"audio_url": url}))["text"] == "你好世界"
    finally:
        httpd.shutdown()
    with pytest.raises(ValueError, match="http"):
        run(tools.call("transcribe_audio", {"audio_url": "file:///etc/passwd"}))


def test_audio_must_come_one_way(tools, wav):
    with pytest.raises(ValueError, match="one of"):
        run(tools.call("transcribe_audio", {}))
    with pytest.raises(ValueError, match="one of"):
        run(tools.call("transcribe_audio", {"audio_path": wav, "audio_base64": _b64_wav()}))


@pytest.fixture
def remote(tools):
    tools.remote, tools.public_url = True, "http://box:8765"
    return tools


def test_remote_mode_returns_a_url_and_base64(remote):
    import base64
    import io

    out = run(remote.call("synthesize_speech", {"text": "你好", "return_base64": True}))
    assert out["url"] == "http://box:8765/files/" + os.path.basename(out["output_path"])
    assert sf.info(io.BytesIO(base64.b64decode(out["audio_base64"]))).samplerate == 24000


def test_remote_mode_keeps_paths_to_its_own_outputs(remote, wav):
    with pytest.raises(PermissionError, match="remote"):
        run(remote.call("transcribe_audio", {"audio_path": wav}))
    with pytest.raises(PermissionError, match="remote"):
        run(remote.call("synthesize_speech", {"text": "x", "output_path": "/tmp/anywhere.wav"}))
    made = run(remote.call("synthesize_speech", {"text": "你好"}))["output_path"]
    assert run(remote.call("transcribe_audio", {"audio_path": made}))["text"] == "你好世界"


def test_http_mcp_round_trip_with_api_key(tools):
    pytest.importorskip("mcp")
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request

    import anyio
    import uvicorn
    from mcp import ClientSession

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tools.remote, tools.public_url = True, f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(mcp_server.create_http_app(tools, api_key="s3cret"),
                                           host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    headers = {"Authorization": "Bearer s3cret"}

    @contextlib.asynccontextmanager
    async def connect():
        try:  # SDK 1.x
            from mcp.client.streamable_http import streamablehttp_client
            async with streamablehttp_client(f"{tools.public_url}/mcp", headers=headers) as streams:
                yield streams
        except ImportError:  # SDK 2.x
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
            async with httpx2.AsyncClient(headers=headers) as http:
                async with streamable_http_client(f"{tools.public_url}/mcp", http_client=http) as streams:
                    yield streams

    async def calls():
        async with connect() as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                result = await session.call_tool("synthesize_speech", {"text": "你好"})
                return names, json.loads(result.content[0].text)

    try:
        names, out = anyio.run(calls)
        assert "transcribe_audio" in names and out["url"].startswith(tools.public_url + "/files/")
        with urllib.request.urlopen(urllib.request.Request(out["url"], headers=headers)) as r:
            assert r.read(4) == b"RIFF"
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(out["url"])
        assert denied.value.code == 401
        with urllib.request.urlopen(f"{tools.public_url}/health") as r:
            assert json.loads(r.read())["status"] == "healthy"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_remote_reads_its_own_result_urls_from_disk(remote, monkeypatch):
    monkeypatch.setattr(mcp_server, "_download", lambda url: pytest.fail("fetched its own file over HTTP"))
    url = run(remote.call("synthesize_speech", {"text": "你好"}))["url"]
    assert run(remote.call("transcribe_audio", {"audio_url": url}))["text"] == "你好世界"
