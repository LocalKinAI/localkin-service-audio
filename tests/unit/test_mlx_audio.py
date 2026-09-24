"""Tests for the mlx-audio strategies, registry wiring and HTTP routing.

mlx-audio is replaced by a fake package, so these run anywhere and download
nothing. They pin down the plumbing each of the ~46 mlx-audio models relies
on: which kwargs reach ``generate``, how languages are spelled, how results
come back.
"""
from __future__ import annotations

import dataclasses
import io
import sys
import types
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localkin_service_audio.core import types as core_types
from localkin_service_audio.core.audio_processing import mlx_audio_support as support
from localkin_service_audio.core.audio_processing import shared
from localkin_service_audio.core.audio_processing.engine import AudioEngine
from localkin_service_audio.core.audio_processing.stt.mlx_audio_strategy import MLXAudioSTTStrategy
from localkin_service_audio.core.audio_processing.tts.mlx_audio_strategy import MLXAudioTTSStrategy
from localkin_service_audio.core.config import model_registry
from localkin_service_audio.core.types import ModelType, TranscriptionResult


# --------------------------------------------------------------------------
# Fake mlx_audio
# --------------------------------------------------------------------------

@dataclass
class FakeSTTOutput:
    text: str
    segments: Optional[list] = None
    language: Optional[str] = None


@dataclass
class FakeGenResult:
    audio: np.ndarray
    sample_rate: int = 24000


class FakeSTTModel:
    def __init__(self):
        self.calls = []

    def generate(self, audio, language=None, verbose=True, max_tokens=None):
        self.calls.append({"audio": audio, "language": language, "max_tokens": max_tokens})
        return FakeSTTOutput(
            text=" 你好世界 ",
            segments=[{"start_time": 0.0, "end_time": 1.2, "text": "你好世界"}],
            language="Chinese",
        )


class FakeTTSModel:
    sample_rate = 24000

    def __init__(self, n_samples=2400):
        self.calls = []
        self.n_samples = n_samples

    # No `speed` parameter on purpose: call_supported must not pass it.
    def generate(self, text, voice=None, lang_code=None, ref_audio=None, ref_text=None, verbose=True):
        self.calls.append(dict(text=text, voice=voice, lang_code=lang_code,
                               ref_audio=ref_audio, ref_text=ref_text))
        if self.n_samples:
            yield FakeGenResult(audio=np.ones(self.n_samples, dtype=np.float32))
            yield FakeGenResult(audio=np.ones((1, self.n_samples), dtype=np.float32))


@pytest.fixture
def fake_mlx(monkeypatch):
    """Install a fake mlx_audio package and pretend to be on Apple Silicon."""
    state = {"stt": FakeSTTModel(), "tts": FakeTTSModel(), "loaded": []}

    def stt_load(repo):
        state["loaded"].append(repo)
        return state["stt"]

    def tts_load(repo):
        state["loaded"].append(repo)
        return state["tts"]

    def load_audio(path, sample_rate=None):
        return f"<audio {path} @ {sample_rate}>"

    pkg = types.ModuleType("mlx_audio")
    stt_utils = types.SimpleNamespace(load=stt_load)
    tts_utils = types.SimpleNamespace(load_model=tts_load)
    monkeypatch.setitem(sys.modules, "mlx_audio", pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", stt_utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", tts_utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", types.SimpleNamespace(load_audio=load_audio))
    monkeypatch.setattr(support, "is_apple_silicon", lambda: True)
    return state


def _mlx_config(name):
    """The catalog entry with its mlx backend applied, whatever this machine is."""
    cfg = model_registry.get(name)
    return dataclasses.replace(cfg, backends={}, backend="mlx", **cfg.backends["mlx"])


def _stt(name, fake_mlx):
    s = MLXAudioSTTStrategy()
    assert s.load(_mlx_config(name)), getattr(s, "load_error", None)
    return s


def _tts(name, fake_mlx):
    s = MLXAudioTTSStrategy()
    assert s.load(_mlx_config(name)), getattr(s, "load_error", None)
    return s


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# Music engines are driven by MusicEngine, not AudioEngine.
_MUSIC_ENGINES = {"heartmula"}


def test_every_registry_model_has_an_implementation():
    """No more names in `kin audio models` that can't load — on any backend.

    v2.0.10 listed twelve models (Parakeet, Canary, Orpheus, Qwen3-TTS, Dia,
    ...) whose engines had no strategy at all.
    """
    engine = AudioEngine()
    missing = []
    for m in model_registry.list_all():
        get = engine._get_stt_strategy_class if m.type == ModelType.STT else engine._get_tts_strategy_class
        engines = {b["engine"] for b in m.backends.values()} if m.backends else {m.engine}
        for e in engines - _MUSIC_ENGINES:
            if get(e) is None:
                missing.append(f"{m.name} ({e})")
    assert not missing, f"registered without a strategy: {missing}"


def test_catalog_entries_are_complete():
    from localkin_service_audio.core.audio_processing.isolated import WORKER_ENVS, WORKERS_DIR

    catalog = [m for m in model_registry.list_all() if m.backends]
    assert len(catalog) >= 45
    assert sum("torch" in m.backends for m in catalog) >= 30
    for m in catalog:
        assert m.languages and m.description, m.name
        for name, b in m.backends.items():
            assert b["repo_id"] and "/" in b["repo_id"], (m.name, name)
            if name == "torch":
                worker = b["parameters"]["worker"]
                assert worker in WORKER_ENVS, (m.name, worker)
                assert (WORKERS_DIR / f"{worker}.py").exists(), worker


def test_engine_resolves_through_registry(monkeypatch):
    engine = AudioEngine()
    monkeypatch.setenv("LOCALKIN_BACKEND", "torch")
    # The prefix is "cosyvoice2"; the model runs in an isolated worker.
    cfg = engine._resolve_config("cosyvoice2:0.5b", ModelType.TTS)
    assert cfg.engine == "isolated" and cfg.repo_id == "FunAudioLLM/CosyVoice2-0.5B"
    assert cfg.parameters["worker"] == "cosyvoice"
    # Names outside the registry still parse as engine:size.
    cfg = engine._resolve_config("faster-whisper:small", ModelType.STT)
    assert (cfg.engine, cfg.model_size) == ("faster-whisper", "small")


@pytest.mark.parametrize("forced,mlx_ok,expected", [
    ("", True, "mlx"), ("", False, "torch"), ("torch", True, "torch"), ("mlx", False, "mlx"),
])
def test_backend_selection(monkeypatch, forced, mlx_ok, expected):
    from localkin_service_audio.core.config import backends

    monkeypatch.setenv("LOCALKIN_BACKEND", forced)
    monkeypatch.setattr(backends, "mlx_available", lambda: mlx_ok)
    cfg = backends.resolve_backend(model_registry.get("qwen3-asr:1.7b"))
    assert cfg.backend == expected
    assert cfg.engine == {"mlx": "mlx-audio", "torch": "isolated"}[expected]
    assert not cfg.backends


def test_single_backend_model_ignores_preference(monkeypatch):
    from localkin_service_audio.core.config import backends

    monkeypatch.setenv("LOCALKIN_BACKEND", "mlx")
    cfg = backends.resolve_backend(model_registry.get("cosyvoice3:0.5b"))
    assert cfg.backend == "torch"  # the only one it has
    assert backends.resolve_backend(model_registry.get("kokoro")).engine == "kokoro"


# --------------------------------------------------------------------------
# Support helpers
# --------------------------------------------------------------------------

def test_call_supported_drops_unknown_and_none_kwargs():
    def fn(text, voice=None, **kwargs):
        return {"text": text, "voice": voice, **kwargs}

    out = shared.call_supported(fn, text="hi", voice=None, speed=1.2, lang_code="en")
    # speed/lang_code aren't named, **kwargs doesn't count; voice=None is dropped.
    assert out == {"text": "hi", "voice": None}


@pytest.mark.parametrize("text,lang", [
    ("你好", "zh"), ("こんにちは世界", "ja"), ("안녕하세요", "ko"), ("hello", "en"),
])
def test_detect_text_language(text, lang):
    assert shared.detect_text_language(text) == lang


def test_resolve_language_prefers_caller_then_text():
    lang_map = {"zh": "chinese", "en": "english", "default": "auto", "auto": "auto"}
    assert shared.resolve_language(None, lang_map, "今天") == "chinese"
    assert shared.resolve_language("en", lang_map, "今天") == "english"
    assert shared.resolve_language("fi", lang_map, "moi") == "auto"
    assert shared.resolve_language(None, None, "今天") is None


def test_load_fails_cleanly_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(support, "is_apple_silicon", lambda: False)
    s = MLXAudioSTTStrategy()
    assert s.load(_mlx_config("qwen3-asr:0.6b")) is False
    assert "Apple Silicon" in s.load_error


# --------------------------------------------------------------------------
# STT strategy
# --------------------------------------------------------------------------

def test_stt_maps_language_names_and_segments(fake_mlx, tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    s = _stt("qwen3-asr:0.6b", fake_mlx)

    r = s.transcribe(str(wav), language="zh")

    assert fake_mlx["loaded"] == ["mlx-community/Qwen3-ASR-0.6B-8bit"]
    assert fake_mlx["stt"].calls[-1]["language"] == "Chinese"  # language_names
    assert r.text == "你好世界"
    assert r.language == "zh"  # "Chinese" mapped back to ISO
    assert r.duration == pytest.approx(1.0)
    assert [(g.start, g.end, g.text) for g in r.segments] == [(0.0, 1.2, "你好世界")]
    assert r.engine == "mlx-audio"


def test_stt_passes_iso_codes_and_registry_defaults(fake_mlx):
    s = _stt("vibevoice-asr:9b", fake_mlx)
    s.transcribe(np.zeros(8000, dtype=np.float32), language="en")
    call = fake_mlx["stt"].calls[-1]
    assert call["language"] == "en"         # no language_names on this entry
    assert call["max_tokens"] == 8192       # parameters.defaults
    assert call["audio"].endswith(".wav")   # arrays go through a temp file


# --------------------------------------------------------------------------
# TTS strategy
# --------------------------------------------------------------------------

def test_tts_picks_language_from_text_and_default_voice(fake_mlx):
    s = _tts("qwen3-tts:0.6b", fake_mlx)
    r = s.synthesize("今天天气很好", speed=1.3)

    call = fake_mlx["tts"].calls[-1]
    assert call["lang_code"] == "chinese"
    assert call["voice"] == "vivian"
    assert r.voice == "vivian"
    assert r.sample_rate == 24000
    # both chunks (4800 samples), then stretched to 1.3x in-process: Qwen3-TTS
    # ignores a speed argument, so none is passed
    assert abs(len(r.audio) - 4800 / 1.3) <= 100
    assert r.duration == pytest.approx(len(r.audio) / 24000)
    assert "speed" not in call


def test_tts_caller_language_and_voice_win(fake_mlx):
    s = _tts("qwen3-tts:0.6b", fake_mlx)
    s.synthesize("今天", voice="uncle_fu", language="en")
    call = fake_mlx["tts"].calls[-1]
    assert (call["voice"], call["lang_code"]) == ("uncle_fu", "english")


def test_tts_clone_only_model_gets_bundled_reference(fake_mlx):
    s = _tts("moss-tts:nano", fake_mlx)
    s.synthesize("你好")
    call = fake_mlx["tts"].calls[-1]
    assert "default_zh.wav" in call["ref_audio"]
    assert call["ref_text"].startswith("你好，很高兴认识你")


def test_tts_caller_reference_overrides_default(fake_mlx, tmp_path):
    s = _tts("moss-tts:nano", fake_mlx)
    s.clone_voice(str(tmp_path / "me.wav"), "hello", "my words")
    call = fake_mlx["tts"].calls[-1]
    assert "me.wav" in call["ref_audio"] and call["ref_text"] == "my words"


def test_tts_empty_audio_is_an_error(fake_mlx):
    fake_mlx["tts"].n_samples = 0
    s = _tts("qwen3-tts:0.6b", fake_mlx)
    with pytest.raises(RuntimeError, match="produced no audio"):
        s.synthesize("hi")


def test_bundled_reference_voices_ship_with_package():
    for tag in ("zh", "en"):
        wav, text = shared.default_reference(tag)
        info = sf.info(wav)
        assert info.duration > 3 and text


# --------------------------------------------------------------------------
# HTTP: any strategy-backed model is served
# --------------------------------------------------------------------------

def test_server_synthesize_routes_mlx_model_and_language(monkeypatch):
    from localkin_service_audio.api import server as srv

    strategy = types.SimpleNamespace(calls=[])

    def synthesize(text, voice=None, speed=1.0, **kw):
        strategy.calls.append(dict(text=text, voice=voice, speed=speed, **kw))
        return core_types.AudioResult(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000)

    strategy.synthesize = synthesize
    monkeypatch.setattr(srv, "loaded_models", {"qwen3-tts:0.6b": {"type": "mlx-audio", "strategy": strategy}})
    client = TestClient(srv.create_app("qwen3-tts:0.6b"))

    r = client.post("/synthesize", json={"text": "你好", "language": "zh", "instruct": "用开心的语气"})
    assert r.status_code == 200, r.text
    assert sf.read(io.BytesIO(r.content))[1] == 24000
    assert strategy.calls[-1]["language"] == "zh"
    assert strategy.calls[-1]["instruct"] == "用开心的语气"


def test_server_transcribe_routes_mlx_model(monkeypatch):
    from localkin_service_audio.api import server as srv

    class Strategy:
        def transcribe(self, path, language=None):
            return TranscriptionResult(text="你好世界", language="zh", duration=1.0)

    monkeypatch.setattr(srv, "loaded_models", {"qwen3-asr:0.6b": {"type": "mlx-audio", "strategy": Strategy()}})
    client = TestClient(srv.create_app("qwen3-asr:0.6b"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")

    r = client.post("/transcribe", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "你好世界"


@pytest.mark.parametrize("device,ram,vram", [
    ("cuda", 32, 24), ("cuda", 32, 8), ("cuda", 16, 4),
    ("mps", 64, 0), ("mps", 16, 0), ("mps", 8, 0),
    ("cpu", 32, 0), ("cpu", 8, 0),
])
def test_recommendations_name_loadable_models(device, ram, vram):
    """`kin audio recommend` used to suggest canary:1b and gpt-sovits,
    neither of which could load."""
    from localkin_service_audio.cli.utils.device import recommend_models
    from localkin_service_audio.core.types import DeviceType, HardwareProfile

    rec = recommend_models(HardwareProfile(device=DeviceType(device), ram_gb=ram, vram_gb=vram, gpu_name="x"))
    engine = AudioEngine()
    for kind, get in (("stt", engine._get_stt_strategy_class), ("tts", engine._get_tts_strategy_class)):
        model_type = ModelType.STT if kind == "stt" else ModelType.TTS
        for name in rec[kind]:
            cfg = engine._resolve_config(name, model_type)
            assert get(cfg.engine) is not None, f"{name} ({cfg.engine})"


@pytest.mark.parametrize("requested,expected", [
    (None, "vivian"), ("uncle_fu", "uncle_fu"), ("Ryan", "ryan"),
    ("zf_xiaoxiao", "vivian"), ("zm_yunxi", "uncle_fu"), ("af_heart", "serena"),
    ("alloy", "vivian"),  # OpenAI name via /v1/audio/speech -> default
])
def test_kokoro_voice_ids_map_onto_qwen3_speakers(fake_mlx, requested, expected):
    """KinClaw sends Kokoro voice ids; Qwen3-TTS must get a speaker it has."""
    s = _tts("qwen3-tts:0.6b", fake_mlx)
    s.synthesize("你好", voice=requested)
    assert fake_mlx["tts"].calls[-1]["voice"] == expected


def test_pick_voice_passes_through_without_voice_list():
    assert shared.pick_voice("anything", None) == "anything"
    assert shared.pick_voice(None, None, default="d") == "d"


def test_transcribe_adds_emotion_from_sidecar(monkeypatch):
    from localkin_service_audio.api import server as srv

    class Main:
        def transcribe(self, path, language=None):
            return TranscriptionResult(text="今天好累", language="zh", duration=1.0)

    class SenseVoice:
        def transcribe(self, path, language=None):
            return TranscriptionResult(text="今天好累", emotion="sad", audio_events=["Laughter"])

    monkeypatch.setattr(srv, "loaded_models", {
        "qwen3-asr:1.7b": {"type": "mlx-audio", "strategy": Main()},
        "sensevoice:small": {"type": "sensevoice", "strategy": SenseVoice()},
    })
    client = TestClient(srv.create_app("qwen3-asr:1.7b", emotion_model="sensevoice:small"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    body = client.post("/transcribe", files={"file": ("a.wav", buf.getvalue(), "audio/wav")}).json()
    assert body["text"] == "今天好累"               # from the main model
    assert body["emotion"] == "sad"                 # from the sidecar
    assert body["audio_events"] == ["Laughter"]
    assert client.get("/health").json()["emotion_model"] == "sensevoice:small"


def test_emotion_sidecar_failure_does_not_fail_transcription(monkeypatch):
    from localkin_service_audio.api import server as srv

    class Main:
        def transcribe(self, path, language=None):
            return TranscriptionResult(text="hi", language="en", duration=1.0)

    def broken_load(name):
        raise RuntimeError("funasr missing")

    monkeypatch.setattr(srv, "loaded_models", {"qwen3-asr:1.7b": {"type": "mlx-audio", "strategy": Main()}})
    monkeypatch.setattr(srv, "load_whisper_model", broken_load)
    client = TestClient(srv.create_app("qwen3-asr:1.7b", emotion_model="sensevoice:small"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    r = client.post("/transcribe", files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and r.json()["text"] == "hi" and "emotion" not in r.json()


def test_preload_loads_before_serving(monkeypatch):
    from localkin_service_audio.api import server as srv

    loaded = []
    monkeypatch.setattr(srv, "load_whisper_model", loaded.append)
    monkeypatch.setattr(srv, "load_tts_model", loaded.append)
    srv.create_app("qwen3-asr:1.7b", emotion_model="sensevoice:small", preload=True)
    srv.create_app("qwen3-tts:0.6b", preload=True)
    srv.create_app("kokoro")  # default: lazy
    assert loaded == ["qwen3-asr:1.7b", "sensevoice:small", "qwen3-tts:0.6b"]


def test_voices_endpoint_lists_qwen3_speakers_as_multilingual(monkeypatch, fake_mlx):
    """KinClaw's voice picker reads this instead of a hard-coded Kokoro list."""
    from localkin_service_audio.api import server as srv

    monkeypatch.setattr(srv, "loaded_models", {"qwen3-tts:0.6b": {"type": "mlx-audio",
                                                                  "strategy": _tts("qwen3-tts:0.6b", fake_mlx)}})
    monkeypatch.setenv("LOCALKIN_BACKEND", "mlx")
    body = TestClient(srv.create_app("qwen3-tts:0.6b")).get("/voices").json()
    assert body["multilingual"] is True
    assert body["default_voice"] == "vivian"
    ids = {v["id"]: v for v in body["voices"]}
    assert set(ids) == {"vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"}
    assert ids["dylan"]["language"] == "zh" and ids["dylan"]["gender"] == "male" and "北京" in ids["dylan"]["name"]


def test_voices_endpoint_marks_kokoro_single_language(monkeypatch):
    from localkin_service_audio.api import server as srv
    from localkin_service_audio.core.audio_processing.tts.kokoro_strategy import KokoroStrategy

    strategy = KokoroStrategy()
    monkeypatch.setattr(srv, "loaded_models", {"kokoro": {"type": "kokoro", "strategy": strategy}})
    body = TestClient(srv.create_app("kokoro")).get("/voices").json()
    assert body["multilingual"] is False
    langs = {v["id"]: v["language"] for v in body["voices"]}
    assert langs["zf_xiaoxiao"] == "zh" and langs["af_heart"] == "en"
