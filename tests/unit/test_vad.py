"""Tests for the engine-agnostic VAD module.

These tests run the actual TEN-VAD backend against synthesized audio to
verify wiring end-to-end (the package ships a native macOS arm64 binary
which is small enough to be a hard test dependency rather than a mock).
"""
from __future__ import annotations

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

ten_vad = pytest.importorskip("ten_vad")  # noqa: F841 — also gates module under test

from localkin_service_audio.core.audio_processing.vad import (  # noqa: E402
    SUPPORTED_VAD_BACKENDS,
    SpeechSegment,
    TenVADBackend,
    detect_speech,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

SR = 16_000


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SR * duration_s), dtype=np.float32)


def _tone(freq_hz: float, duration_s: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Backend smoke tests
# --------------------------------------------------------------------------

class TestTenVADBackend:
    def test_constants_match_ten_vad_defaults(self):
        backend = TenVADBackend()
        assert backend.SAMPLE_RATE == 16_000
        assert backend.HOP_SIZE == 256
        assert backend.threshold == 0.5

    def test_empty_audio_returns_no_segments(self):
        backend = TenVADBackend()
        segments = backend.detect_speech(_silence(0.5))
        assert segments == []

    def test_segments_have_correct_dataclass_shape(self):
        # Real-ish signal with at least one speech-like burst.
        audio = np.concatenate([
            _silence(0.5),
            _tone(440, 1.0),
            _silence(0.5),
        ])
        segments = detect_speech(audio, threshold=0.3)
        # The exact segmentation depends on ten-vad's heuristics, but the
        # contract we test is the dataclass shape.
        for seg in segments:
            assert isinstance(seg, SpeechSegment)
            assert seg.end > seg.start
            assert seg.duration == pytest.approx(seg.end - seg.start)

    def test_detect_speech_accepts_file_path(self, tmp_path):
        audio = np.concatenate([_silence(0.3), _tone(440, 0.8), _silence(0.3)])
        path = tmp_path / "test.wav"
        sf.write(path, audio, SR)
        # Just verify it doesn't crash and returns the same type.
        segs = detect_speech(str(path), threshold=0.3)
        assert isinstance(segs, list)

    def test_min_speech_duration_drops_short_blips(self):
        # 2 frames @ 16ms each = 32ms speech, well below 200ms threshold.
        audio = np.concatenate([
            _silence(0.5),
            _tone(440, 0.03),  # 30ms
            _silence(0.5),
        ])
        segments = detect_speech(audio, threshold=0.3, min_speech_duration_ms=200)
        # Either zero segments OR no segment shorter than 200ms.
        for seg in segments:
            assert seg.duration >= 0.2 - 0.05  # allow 50ms slack for padding

    def test_int16_input_works(self):
        backend = TenVADBackend()
        audio = (_tone(440, 1.0) * 32767).astype(np.int16)
        # Should not raise.
        segments = backend.detect_speech(audio)
        assert isinstance(segments, list)


# --------------------------------------------------------------------------
# Top-level helper
# --------------------------------------------------------------------------

class TestDetectSpeechHelper:
    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="Unsupported VAD backend"):
            detect_speech(_silence(0.1), backend="silero")

    def test_supported_backends_constant_includes_ten_vad(self):
        assert "ten-vad" in SUPPORTED_VAD_BACKENDS


# --------------------------------------------------------------------------
# /vad HTTP endpoint
# --------------------------------------------------------------------------

class TestVADEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        """A TestClient on a server bound to a fake STT model."""
        from localkin_service_audio.api import server as srv

        model_name = "faster-whisper:base"
        monkeypatch.setattr(srv, "loaded_models", {
            model_name: {"type": "faster-whisper", "strategy": MagicMock()},
        })
        monkeypatch.setattr(srv, "load_whisper_model", lambda name: None)

        app = srv.create_app(model_name)
        return TestClient(app)

    def test_invalid_backend_returns_400(self, client):
        wav = _wav_bytes(_silence(0.5))
        r = client.post(
            "/vad",
            params={"backend": "silero"},
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 400
        assert "ten-vad" in r.json()["detail"]

    def test_silence_returns_empty_segments(self, client):
        wav = _wav_bytes(_silence(0.5))
        r = client.post(
            "/vad",
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["backend"] == "ten-vad"
        assert body["speech_segments"] == []
        assert body["total_speech_duration"] == 0.0
        assert body["duration"] == pytest.approx(0.5, abs=0.05)

    def test_returns_segment_shape(self, client):
        # Use a sustained tone to maximize chance of a detection.
        audio = np.concatenate([_silence(0.5), _tone(440, 1.5, 0.5), _silence(0.5)])
        wav = _wav_bytes(audio)
        r = client.post(
            "/vad",
            params={"threshold": 0.2},
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        # Whatever the backend returns, the shape must match.
        for seg in body["speech_segments"]:
            assert set(seg.keys()) == {"start", "end", "duration"}
            assert seg["end"] >= seg["start"]

    def test_tunable_params_passed_through(self, client, monkeypatch):
        """The endpoint passes our params to ``detect_speech``."""
        from localkin_service_audio.core.audio_processing import vad as vad_mod

        called_with: dict = {}

        def spy(audio, **kwargs):
            called_with["audio"] = audio
            called_with.update(kwargs)
            return []

        monkeypatch.setattr(vad_mod, "detect_speech", spy)

        wav = _wav_bytes(_silence(0.3))
        r = client.post(
            "/vad",
            params={
                "threshold": 0.7,
                "min_speech_duration_ms": 300,
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 50,
            },
            files={"file": ("test.wav", wav, "audio/wav")},
        )
        assert r.status_code == 200
        assert called_with["backend"] == "ten-vad"
        assert called_with["threshold"] == 0.7
        assert called_with["min_speech_duration_ms"] == 300
        assert called_with["min_silence_duration_ms"] == 400
        assert called_with["speech_pad_ms"] == 50
