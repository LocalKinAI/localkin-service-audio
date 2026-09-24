"""HTTP tests for /synthesize (Kokoro), /v1/audio/speech and /health.

Model loading is stubbed out, as in test_transcribe_endpoint.py, so these
run without weights. The Kokoro strategy tests fake the ``kokoro`` package
to check which pipeline and voice a request actually reaches.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localkin_service_audio.api import server as srv
from localkin_service_audio.core.audio_processing.tts.kokoro_strategy import KokoroStrategy
from localkin_service_audio.core.types import AudioResult, ModelConfig, ModelType


# --------------------------------------------------------------------------
# /synthesize goes through KokoroStrategy
# --------------------------------------------------------------------------

@pytest.fixture
def tts_client(monkeypatch):
    strategy = MagicMock()
    strategy.synthesize.return_value = AudioResult(
        audio=np.zeros(24000, dtype=np.float32), sample_rate=24000, model="kokoro",
    )
    monkeypatch.setattr(srv, "loaded_models", {
        "kokoro": {"type": "kokoro", "strategy": strategy},
    })
    monkeypatch.setattr(srv, "load_tts_model", lambda name: None)
    return TestClient(srv.create_app("kokoro")), strategy


def test_synthesize_without_speaker_lets_strategy_pick_voice(tts_client):
    client, strategy = tts_client
    r = client.post("/synthesize", json={"text": "你好，世界"})

    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    audio, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and len(audio) == 24000
    # No hard-coded af_heart: the voice is left to the strategy.
    _, kwargs = strategy.synthesize.call_args
    assert kwargs["voice"] is None


def test_synthesize_passes_speaker_and_clamps_speed(tts_client):
    client, strategy = tts_client
    r = client.post("/synthesize", json={"text": "hi", "speaker": "am_adam", "speed": 9})

    assert r.status_code == 200
    _, kwargs = strategy.synthesize.call_args
    assert kwargs["voice"] == "am_adam"
    assert kwargs["speed"] == 2.0


def test_synthesize_empty_audio_is_an_error_not_a_silent_wav(tts_client):
    client, strategy = tts_client
    strategy.synthesize.side_effect = RuntimeError("Kokoro produced no audio for voice 'af_heart'")
    r = client.post("/synthesize", json={"text": "你好", "speaker": "af_heart"})

    assert r.status_code == 500
    # Reported as a synthesis failure, not re-wrapped as a load failure.
    assert r.json()["detail"].startswith("Synthesis failed:")
    assert "af_heart" in r.json()["detail"]


def test_openai_speech_maps_voice_and_speed(tts_client):
    client, strategy = tts_client
    r = client.post("/v1/audio/speech", json={"input": "hello", "voice": "bf_emma", "speed": 1.3})

    assert r.status_code == 200
    args, kwargs = strategy.synthesize.call_args
    assert args[0] == "hello"
    assert kwargs["voice"] == "bf_emma"
    assert kwargs["speed"] == 1.3


# --------------------------------------------------------------------------
# /health reports a missing backend
# --------------------------------------------------------------------------

def test_health_is_503_when_backend_missing(monkeypatch):
    monkeypatch.setattr(srv, "loaded_models", {})
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    r = TestClient(srv.create_app("sensevoice:small")).get("/health")

    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy"
    assert "funasr" in body["error"]


def test_health_ok_when_backend_installed(monkeypatch):
    monkeypatch.setattr(srv, "loaded_models", {})
    monkeypatch.setattr(srv, "_missing_backend", lambda info: None)
    r = TestClient(srv.create_app("sensevoice:small")).get("/health")

    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "model": "sensevoice:small", "loaded": False}


def test_missing_backend_accepts_any_alternative(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name: object() if name == "moonshine" else None)
    assert srv._missing_backend({"engine": "moonshine"}) is None
    assert srv._missing_backend({"engine": "kokoro"}) == "kokoro"
    # Engines without a listed module (transformers-based) are not checked.
    assert srv._missing_backend({"engine": "chattts"}) is None


def test_missing_backend_for_isolated_models_is_uv(monkeypatch):
    from localkin_service_audio.core.audio_processing import isolated
    monkeypatch.setattr(isolated, "find_uv", lambda: None)
    assert "uv" in srv._missing_backend({"engine": "isolated"})
    monkeypatch.setattr(isolated, "find_uv", lambda: "/usr/bin/uv")
    assert srv._missing_backend({"engine": "isolated"}) is None


# --------------------------------------------------------------------------
# KokoroStrategy picks the voice from the text when none is given
# --------------------------------------------------------------------------

@pytest.fixture
def fake_kokoro(monkeypatch):
    """Install a fake ``kokoro`` module recording each pipeline call."""
    calls = []

    class KPipeline:
        def __init__(self, lang_code):
            self.lang_code = lang_code

        def __call__(self, text, voice, speed):
            calls.append((self.lang_code, voice, speed))
            yield None, None, np.ones(2400, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=KPipeline))
    monkeypatch.setattr(KokoroStrategy, "_ensure_spacy_model", staticmethod(lambda: None))
    strategy = KokoroStrategy()
    assert strategy.load(ModelConfig(name="kokoro", type=ModelType.TTS, engine="kokoro"))
    return strategy, calls


@pytest.mark.parametrize("text,lang,voice", [
    ("你好，世界", "z", "zf_xiaoxiao"),
    ("こんにちは世界", "j", "jf_alpha"),
    ("Hello world", "a", "af_heart"),
])
def test_kokoro_default_voice_follows_text_language(fake_kokoro, text, lang, voice):
    strategy, calls = fake_kokoro
    result = strategy.synthesize(text)

    assert calls == [(lang, voice, 1.0)]
    assert result.voice == voice


def test_kokoro_explicit_voice_wins(fake_kokoro):
    strategy, calls = fake_kokoro
    strategy.synthesize("你好", voice="zm_yunxi", speed=1.2)
    assert calls == [("z", "zm_yunxi", 1.2)]


def test_sensevoice_loads_without_remote_code(monkeypatch):
    """trust_remote_code made funasr `pip install -r` the model repo's
    requirements (numpy<=1.26.4, gradio) into whatever pip was on PATH."""
    from localkin_service_audio.core.audio_processing.stt.sensevoice_strategy import SenseVoiceStrategy

    seen = {}

    class FakeAutoModel:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))
    s = SenseVoiceStrategy()
    assert s.load(ModelConfig(name="sensevoice:small", type=ModelType.STT, engine="sensevoice"))
    assert "trust_remote_code" not in seen and "remote_code" not in seen
    assert seen["model"] == "FunAudioLLM/SenseVoiceSmall"


def test_kokoro_setup_exit_does_not_kill_the_process(monkeypatch):
    """spacy.cli.download sys.exit()s when it can't install; load() must
    report that instead of letting SystemExit take the server down."""
    def exiting_pipeline(lang_code):
        raise SystemExit(1)

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=exiting_pipeline))
    monkeypatch.setattr(KokoroStrategy, "_ensure_spacy_model", staticmethod(lambda: None))
    s = KokoroStrategy()
    assert s.load(ModelConfig(name="kokoro", type=ModelType.TTS, engine="kokoro")) is False
    assert "spaCy" in s.load_error


def test_kokoro_installs_spacy_model_without_pip_or_uv(monkeypatch):
    import shutil
    import subprocess

    fake_spacy = types.SimpleNamespace(__version__="3.8.7",
                                       util=types.SimpleNamespace(is_package=lambda name: False))
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    monkeypatch.setitem(sys.modules, "spacy.util", fake_spacy.util)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def no_pip(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "check_call", no_pip)
    unpacked = []
    monkeypatch.setattr(KokoroStrategy, "_unpack_wheel", staticmethod(unpacked.append))
    KokoroStrategy._ensure_spacy_model()
    assert unpacked and unpacked[0].endswith("en_core_web_sm-3.8.0-py3-none-any.whl")
