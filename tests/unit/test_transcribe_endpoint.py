"""HTTP integration tests for the /transcribe endpoint.

These tests use FastAPI's TestClient against a real ``create_app(...)``
instance, but stub out the actual model load so they run instantly and
need no model weights. They validate the new transcription-control
parameters added in v2.0.11 (resolves #2):

  - enable_vad
  - timestamps
  - response_format (json | text | markdown | srt | vtt)
  - chunk_length_s

and verify that the JSON shape stays back-compatible for callers that
don't pass any new parameters.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pytest
import soundfile as sf
import numpy as np
from fastapi.testclient import TestClient

from localkin_service_audio.api import server as srv


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000) -> bytes:
    """Generate a small in-memory WAV file (silence is fine for these tests)."""
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def stub_strategy():
    """A strategy mock that returns a deterministic 2-segment result."""
    strategy = MagicMock()
    fake_segments = [
        MagicMock(start=0.0, end=1.5, text="Hello world."),
        MagicMock(start=1.5, end=3.0, text="This is a test."),
    ]
    strategy.transcribe.return_value = MagicMock(
        text="Hello world. This is a test.",
        language="en",
        duration=3.0,
        segments=fake_segments,
    )
    return strategy


@pytest.fixture
def client(monkeypatch, stub_strategy):
    """A TestClient with a fake-loaded faster-whisper model.

    Bypasses real model loading by pre-seeding ``loaded_models`` and
    short-circuiting ``load_whisper_model``.
    """
    model_name = "faster-whisper:base"

    # Reset the global cache between tests to avoid cross-test bleed.
    monkeypatch.setattr(srv, "loaded_models", {
        model_name: {"type": "faster-whisper", "strategy": stub_strategy},
    })
    monkeypatch.setattr(srv, "load_whisper_model", lambda name: None)

    app = srv.create_app(model_name)
    return TestClient(app), stub_strategy, model_name


def _post_transcribe(test_client: TestClient, params: dict | None = None):
    """Helper to POST a small WAV file with optional query params."""
    files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
    return test_client.post("/transcribe", params=params or {}, files=files)


# --------------------------------------------------------------------------
# Back-compat: default JSON shape stays exactly as v2.0.10
# --------------------------------------------------------------------------

class TestBackwardCompat:
    def test_default_json_shape_unchanged(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc)
        assert r.status_code == 200, r.text
        body = r.json()
        # Must contain the v2.0.x fields and NOT contain segments by default.
        assert "text" in body
        assert "language" in body
        assert "segments" not in body
        assert body["text"] == "Hello world. This is a test."
        assert body["language"] == "en"

    def test_default_passes_vad_true_to_strategy(self, client):
        tc, strategy, _ = client
        _post_transcribe(tc)
        kwargs = strategy.transcribe.call_args.kwargs
        # Default enable_vad=True; chunk_length not forwarded when None.
        assert kwargs.get("enable_vad") is True
        assert "chunk_length" not in kwargs


# --------------------------------------------------------------------------
# enable_vad parameter
# --------------------------------------------------------------------------

class TestEnableVAD:
    def test_disable_vad_passes_through(self, client):
        tc, strategy, _ = client
        r = _post_transcribe(tc, {"enable_vad": "false"})
        assert r.status_code == 200
        assert strategy.transcribe.call_args.kwargs["enable_vad"] is False

    def test_explicit_true_still_works(self, client):
        tc, strategy, _ = client
        _post_transcribe(tc, {"enable_vad": "true"})
        assert strategy.transcribe.call_args.kwargs["enable_vad"] is True


# --------------------------------------------------------------------------
# timestamps parameter (JSON shape)
# --------------------------------------------------------------------------

class TestTimestamps:
    def test_timestamps_true_includes_segments(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"timestamps": "true"})
        assert r.status_code == 200
        body = r.json()
        assert "segments" in body
        assert len(body["segments"]) == 2
        assert body["segments"][0] == {
            "start": 0.0,
            "end": 1.5,
            "text": "Hello world.",
        }

    def test_timestamps_false_omits_segments(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"timestamps": "false"})
        assert "segments" not in r.json()


# --------------------------------------------------------------------------
# response_format parameter
# --------------------------------------------------------------------------

class TestResponseFormat:
    def test_text_format(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"response_format": "text"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text.strip() == "Hello world. This is a test."

    def test_markdown_format_has_metadata_and_segments(self, client):
        tc, _, model_name = client
        r = _post_transcribe(tc, {"response_format": "markdown"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/markdown")
        body = r.text
        assert "# Transcription" in body
        assert "**Language:** en" in body
        assert f"**Model:** `{model_name}`" in body
        assert "## Segments" in body
        assert "**[00:00 → 00:01]** Hello world." in body
        assert "**[00:01 → 00:03]** This is a test." in body

    def test_srt_format(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"response_format": "srt"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-subrip")
        body = r.text
        assert "1\n00:00:00,000 --> 00:00:01,500\nHello world." in body
        assert "2\n00:00:01,500 --> 00:00:03,000\nThis is a test." in body

    def test_vtt_format(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"response_format": "vtt"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/vtt")
        body = r.text
        assert body.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.500" in body

    def test_invalid_format_returns_400(self, client):
        tc, _, _ = client
        r = _post_transcribe(tc, {"response_format": "html"})
        assert r.status_code == 400
        assert "response_format" in r.json()["detail"]

    def test_subtitle_without_segments_returns_422(self, monkeypatch, client):
        """SRT/VTT need timestamps — fail loudly if engine returns none."""
        tc, strategy, _ = client
        # Make the engine return no segments.
        strategy.transcribe.return_value = MagicMock(
            text="hi", language="en", duration=None, segments=None,
        )
        r = _post_transcribe(tc, {"response_format": "srt"})
        assert r.status_code == 422
        assert "segment timestamps" in r.json()["detail"]


# --------------------------------------------------------------------------
# chunk_length_s parameter
# --------------------------------------------------------------------------

class TestChunkLength:
    def test_chunk_length_forwarded_to_faster_whisper(self, client):
        tc, strategy, _ = client
        _post_transcribe(tc, {"chunk_length_s": 15})
        kwargs = strategy.transcribe.call_args.kwargs
        # _run_stt maps API param chunk_length_s → strategy kwarg chunk_length.
        assert kwargs.get("chunk_length") == 15

    def test_no_chunk_length_means_kwarg_omitted(self, client):
        tc, strategy, _ = client
        _post_transcribe(tc)
        assert "chunk_length" not in strategy.transcribe.call_args.kwargs


# --------------------------------------------------------------------------
# OpenAI-compat endpoint mirrors /transcribe behavior
# --------------------------------------------------------------------------

class TestOpenAICompat:
    def test_openai_endpoint_accepts_new_params(self, client):
        tc, strategy, _ = client
        files = {"file": ("test.wav", _make_wav_bytes(), "audio/wav")}
        r = tc.post(
            "/v1/audio/transcriptions",
            params={"response_format": "markdown", "enable_vad": "false"},
            files=files,
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/markdown")
        assert strategy.transcribe.call_args.kwargs["enable_vad"] is False
