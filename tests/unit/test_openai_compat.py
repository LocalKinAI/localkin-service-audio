"""OpenAI-compatible endpoints and the combined STT+TTS server.

What an OpenAI SDK actually sends: options as multipart form fields on
/v1/audio/transcriptions, a JSON body with ``instructions`` and
``response_format`` on /v1/audio/speech. Models are stubbed, as in
test_transcribe_endpoint.py and test_synthesize_endpoint.py.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from localkin_service_audio.api import server as srv
from localkin_service_audio.core.types import AudioResult

STT, TTS = "faster-whisper:base", "kokoro"


def _wav(seconds=0.5, sr=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def models(monkeypatch):
    stt = MagicMock()
    stt.transcribe.return_value = MagicMock(
        text="你好 世界", language="zh", duration=1.0, emotion=None, audio_events=None,
        segments=[MagicMock(start=0.0, end=0.5, text="你好"), MagicMock(start=0.5, end=1.0, text="世界")],
    )
    tts = MagicMock()
    tts.synthesize.return_value = AudioResult(audio=np.full(16000, 0.1, dtype=np.float32), sample_rate=16000)
    monkeypatch.setattr(srv, "loaded_models", {
        STT: {"type": "faster-whisper", "strategy": stt},
        TTS: {"type": "kokoro", "strategy": tts},
    })
    monkeypatch.setattr(srv, "load_whisper_model", lambda name: None)
    monkeypatch.setattr(srv, "load_tts_model", lambda name: None)
    return stt, tts


def _pieces(tts, n=2, sr=16000):
    tts.synthesize_stream.side_effect = lambda text, **kw: iter(
        [AudioResult(audio=np.full(sr // 2, 0.1, dtype=np.float32), sample_rate=sr) for _ in range(n)])


def _transcribe(client, **form):
    return client.post("/v1/audio/transcriptions", data=form,
                       files={"file": ("a.wav", _wav(), "audio/wav")})


def test_transcription_reads_the_sdks_form_fields(models):
    stt, _ = models
    client = TestClient(srv.create_app(STT))
    r = _transcribe(client, model="whisper-1", language="zh", response_format="text")
    assert r.status_code == 200 and r.text == "你好 世界\n"
    assert stt.transcribe.call_args.kwargs["language"] == "zh"


def test_transcription_query_parameters_still_work(models):
    client = TestClient(srv.create_app(STT))
    r = client.post("/v1/audio/transcriptions?response_format=text",
                    files={"file": ("a.wav", _wav(), "audio/wav")})
    assert r.text == "你好 世界\n"


def test_verbose_json_has_openais_shape(models):
    client = TestClient(srv.create_app(STT))
    body = _transcribe(client, response_format="verbose_json").json()
    assert body["task"] == "transcribe" and body["text"] == "你好 世界" and body["duration"] == 1.0
    assert [(s["id"], s["start"], s["text"]) for s in body["segments"]] == [(0, 0.0, "你好"), (1, 0.5, "世界")]


@pytest.mark.parametrize("fmt,media", [("mp3", "audio/mpeg"), ("opus", "audio/ogg"), ("flac", "audio/flac")])
def test_speech_encodes_the_requested_format(models, fmt, media):
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "hi", "voice": "alloy",
                                              "response_format": fmt})
    assert r.status_code == 200 and r.headers["content-type"] == media
    audio, sr = sf.read(io.BytesIO(r.content))
    assert abs(len(audio) / sr - 1.0) < 0.1


def test_speech_pcm_is_16_bit_24_khz_and_streams_by_default(models):
    _, tts = models
    _pieces(tts)
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.headers["content-type"] == "audio/pcm" and tts.synthesize_stream.called
    assert len(r.content) == 24000 * 2  # 1 s of 16 kHz audio, resampled to 24 kHz, 2 bytes a sample


def test_speech_defaults_to_wav_and_maps_instructions(models):
    _, tts = models
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "你好", "instructions": "用开心的语气"})
    assert r.headers["content-type"] == "audio/wav"
    assert tts.synthesize.call_args.kwargs["instruct"] == "用开心的语气"


def test_speech_rejects_an_unknown_format(models):
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "aac"})
    assert r.status_code == 400 and "mp3" in r.json()["detail"]


def test_models_lists_the_served_model(models):
    assert TestClient(srv.create_app(TTS)).get("/v1/models").json()["data"][0]["id"] == TTS


def test_combined_server_answers_both_on_one_port(models):
    stt, tts = models
    client = TestClient(srv.create_combined_app(STT, TTS))
    assert _transcribe(client, response_format="text").text == "你好 世界\n"
    assert client.post("/v1/audio/speech", json={"input": "hi"}).status_code == 200
    assert client.post("/synthesize", json={"text": "hi"}).status_code == 200
    assert client.get("/voices").status_code == 200
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == [STT, TTS]
    health = client.get("/health").json()
    assert health["status"] == "healthy" and health["models"][TTS]["loaded"]
    paths = [getattr(r, "path", None) for r in client.app.router.routes]
    assert paths.count("/health") == 1 and paths.count("/vad") == 1


def test_combined_server_catches_swapped_models(models):
    with pytest.raises(ValueError, match="pass it as --tts"):
        srv.create_combined_app(stt_model=TTS, tts_model=STT)


def test_combined_server_with_one_model_is_that_models_server(models):
    app = srv.create_combined_app(tts_model=TTS)
    assert TestClient(app).get("/health").json()["model"] == TTS


def _mp3(seconds=0.5, sr=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.full(int(sr * seconds), 0.05, dtype=np.float32), sr, format="MP3")
    return buf.getvalue()


def test_uploaded_mp3_reaches_the_model_as_wav(models):
    stt, _ = models
    seen = {}
    stt.transcribe.side_effect = lambda path, **kw: (seen.update(info=sf.info(path)),
                                                     stt.transcribe.return_value)[1]
    client = TestClient(srv.create_app(STT))
    r = client.post("/v1/audio/transcriptions", data={"response_format": "text"},
                    files={"file": ("speech.mp3", _mp3(), "audio/mpeg")})
    assert r.status_code == 200 and seen["info"].format == "WAV"


def test_undecodable_upload_is_a_400(models, monkeypatch):
    import librosa

    monkeypatch.setattr(librosa, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    client = TestClient(srv.create_app(STT))
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.m4a", b"not audio", "audio/mp4")})
    assert r.status_code == 400 and "ffmpeg" in r.json()["detail"]


@pytest.fixture
def keyed(models):
    from localkin_service_audio.api.auth import BearerAuth

    app = srv.create_combined_app(STT, TTS)
    app.add_middleware(BearerAuth, api_key="s3cret")
    return TestClient(app)


def test_api_key_is_required_except_for_health(keyed):
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/v1/models").status_code == 401
    assert keyed.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert keyed.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert keyed.get("/v1/models", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_openai_sdk_api_key_is_the_access_key(keyed):
    r = keyed.post("/v1/audio/speech", json={"input": "hi"}, headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert keyed.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401


def test_speech_streams_pcm_as_it_is_made(models):
    _, tts = models
    _pieces(tts)
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "一句。两句。", "response_format": "pcm",
                                              "stream_format": "audio", "instructions": "温柔"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/pcm"
    assert len(r.content) == 2 * 12000 * 2  # two 0.5 s pieces at 24 kHz, 16-bit
    assert tts.synthesize_stream.call_args.kwargs["instruct"] == "温柔"


def test_speech_streams_wav_with_an_open_ended_header(models):
    _, tts = models
    _pieces(tts)
    client = TestClient(srv.create_app(TTS))
    body = client.post("/v1/audio/speech", json={"input": "x", "response_format": "wav",
                                                 "stream_format": "audio"}).content
    assert body[:4] == b"RIFF" and body[40:44] == b"\xff\xff\xff\xff" and len(body) == 44 + 2 * 8000 * 2


def test_speech_streams_mp3_pieces_that_play_back_to_back(models):
    """One stream of bare frames: no piece's Xing/Info header to stop a decoder halfway."""
    _, tts = models
    _pieces(tts)
    client = TestClient(srv.create_app(TTS))
    body = client.post("/v1/audio/speech", json={"input": "x", "response_format": "mp3",
                                                 "stream_format": "audio"}).content
    offset, frames = 0, 0
    while offset < len(body):
        length = srv._mp3_frame_length(body[offset:offset + 4])
        assert length, f"no frame header at byte {offset}"
        offset, frames = offset + length, frames + 1
    assert offset == len(body) and b"Xing" not in body and b"Info" not in body
    assert frames * 576 >= 2 * 8000  # MPEG-2 Layer III: 576 samples a frame; two 0.5 s pieces at 16 kHz


def test_speech_streams_sse_events(models):
    import base64
    import json

    _, tts = models
    _pieces(tts)
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "x", "response_format": "pcm", "stream_format": "sse"})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert r.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in events] == ["speech.audio.delta", "speech.audio.delta", "speech.audio.done"]
    assert len(base64.b64decode(events[0]["audio"])) == 12000 * 2


def test_stream_that_fails_at_once_is_a_500(models):
    _, tts = models
    tts.synthesize_stream.side_effect = RuntimeError("out of memory")
    client = TestClient(srv.create_app(TTS))
    r = client.post("/v1/audio/speech", json={"input": "x", "response_format": "pcm", "stream_format": "audio"})
    assert r.status_code == 500 and "out of memory" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"stream_format": "chunks"}, {"stream_format": "audio", "response_format": "flac"}])
def test_stream_rejects_what_it_cannot_send(models, body):
    client = TestClient(srv.create_app(TTS))
    assert client.post("/v1/audio/speech", json={"input": "x", **body}).status_code == 400
