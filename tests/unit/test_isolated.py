"""Tests for isolated-environment workers.

A fake worker script speaks the real protocol from workers/_protocol.py in
a real subprocess (this interpreter, no environment build), so the whole
request/reply path runs without installing or downloading anything.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from localkin_service_audio.core.audio_processing import isolated
from localkin_service_audio.core.config import model_registry

FAKE_WORKER = textwrap.dedent('''
    import sys
    from _protocol import serve, write_wav

    state = {}

    def load(repo, params=None):
        import os, subprocess
        # None of these may reach the reply channel.
        print("noisy library output on stdout")
        os.write(1, b"native code writing to fd 1\\n")
        subprocess.run(["echo", "a child process writing to stdout"])
        if repo == "broken/repo":
            raise RuntimeError("weights not found")
        state["repo"] = repo
        return {"device": "cpu"}

    def synthesize(text, out, voice=None, language=None, ref_audio=None, ref_text=None, **extra):
        state["last"] = dict(text=text, voice=voice, language=language,
                             ref_audio=ref_audio, ref_text=ref_text, extra=extra)
        return write_wav(out, [0.1] * 2400, 24000)

    def transcribe(audio, language=None, **_):
        return {"text": " 你好 ", "language": "Chinese",
                "segments": [{"start": 0.0, "end": 0.5, "text": "你好"}]}

    def last():
        return state["last"]

    serve({"load": load, "synthesize": synthesize, "transcribe": transcribe, "last": last})
''')


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    """Point the 'fake' worker at this interpreter and a temp workers dir."""
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "_protocol.py").write_text((isolated.WORKERS_DIR / "_protocol.py").read_text())
    (workers / "fake.py").write_text(FAKE_WORKER)
    monkeypatch.setattr(isolated, "WORKERS_DIR", workers)
    monkeypatch.setitem(isolated.WORKER_ENVS, "fake", "fake")
    monkeypatch.setattr(isolated, "ensure_env", lambda name: {})
    monkeypatch.setattr(isolated, "_python", lambda root: sys.executable)
    return workers


def _config(name, **params):
    cfg = model_registry.get(name)
    torch_backend = dict(cfg.backends["torch"])
    torch_backend["parameters"] = {**torch_backend["parameters"], "worker": "fake", **params}
    return dataclasses.replace(cfg, backends={}, backend="torch", **torch_backend)


def test_tts_round_trip_through_worker(fake_env):
    s = isolated.IsolatedTTSStrategy()
    assert s.load(_config("qwen3-tts:0.6b")), getattr(s, "load_error", None)
    try:
        r = s.synthesize("今天天气很好", instruct="开心")
        assert r.sample_rate == 24000 and len(r.audio) == 2400
        assert r.voice == "Vivian"  # torch backend default
        sent = s.worker.call("last")
        assert sent["language"] == "zh"  # detected from the text
        assert sent["extra"]["instruct"] == "开心"
        assert sent["ref_audio"] is None
    finally:
        s.unload()


def test_speed_is_applied_by_stretching(fake_env):
    """Workers don't honour speed (Qwen3-TTS ignores it), so it's applied
    after synthesis and never sent to the worker."""
    s = isolated.IsolatedTTSStrategy()
    assert s.load(_config("qwen3-tts:0.6b"))
    try:
        r = s.synthesize("你好", speed=2.0)
        assert abs(len(r.audio) - 1200) <= 100  # 2400 samples at 2x
        assert "speed" not in s.worker.call("last")["extra"]
    finally:
        s.unload()


def test_clone_only_model_sends_bundled_reference(fake_env):
    s = isolated.IsolatedTTSStrategy()
    assert s.load(_config("cosyvoice3:0.5b"))
    try:
        s.synthesize("hello there")
        sent = s.worker.call("last")
        assert sent["ref_audio"].endswith("default_en.wav")
        assert sent["ref_text"].startswith("Hello, it's nice to meet you")
    finally:
        s.unload()


def test_stt_round_trip_through_worker(fake_env, tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(8000, dtype=np.float32), 16000)
    s = isolated.IsolatedSTTStrategy()
    assert s.load(_config("qwen3-asr:0.6b"))
    try:
        r = s.transcribe(str(wav))
        assert (r.text, r.language, r.duration) == ("你好", "zh", 0.5)
        assert [(g.start, g.end, g.text) for g in r.segments] == [(0.0, 0.5, "你好")]
        # arrays go through a temp file
        assert s.transcribe(np.zeros(1600, dtype=np.float32)).text == "你好"
    finally:
        s.unload()


def test_load_error_is_reported_and_worker_stopped(fake_env):
    cfg = dataclasses.replace(_config("qwen3-asr:0.6b"), repo_id="broken/repo")
    s = isolated.IsolatedSTTStrategy()
    assert s.load(cfg) is False
    assert "weights not found" in s.load_error
    assert s.worker.proc.poll() is not None


def test_worker_death_is_an_error_not_a_hang(fake_env):
    s = isolated.IsolatedTTSStrategy()
    assert s.load(_config("qwen3-tts:0.6b"))
    s.worker.proc.kill()
    s.worker.proc.wait()
    with pytest.raises(RuntimeError, match="worker exited"):
        s.synthesize("hi")


def test_ensure_env_builds_once_and_rebuilds_on_spec_change(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("LOCALKIN_HOME", str(tmp_path))
    monkeypatch.setattr(isolated, "find_uv", lambda: "/usr/bin/uv")

    def fake_run(cmd):
        calls.append(cmd)
        if cmd[1] == "venv":
            py = isolated._python(tmp_path / "envs" / "demo")
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")

    monkeypatch.setattr(isolated, "_run", fake_run)
    monkeypatch.setitem(isolated.ENVS, "demo", {"python": "3.11", "packages": ["pkg==1"]})

    isolated.ensure_env("demo")
    isolated.ensure_env("demo")
    assert [c[1] for c in calls] == ["venv", "pip"]
    assert calls[1][-1] == "pkg==1"

    monkeypatch.setitem(isolated.ENVS, "demo", {"python": "3.11", "packages": ["pkg==2"]})
    isolated.ensure_env("demo")
    assert calls[-1][-1] == "pkg==2"


def test_ensure_env_respects_no_auto_install(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALKIN_HOME", str(tmp_path))
    monkeypatch.setenv("LOCALKIN_AUTO_INSTALL", "0")
    monkeypatch.setattr(isolated, "find_uv", lambda: "/usr/bin/uv")
    with pytest.raises(RuntimeError, match="LOCALKIN_AUTO_INSTALL=0"):
        isolated.ensure_env("qwen3_tts")


def test_every_env_spec_is_buildable_shape():
    for name, spec in isolated.ENVS.items():
        assert spec["python"].startswith("3."), name
        assert spec.get("packages") or spec.get("git"), name
    assert set(isolated.WORKER_ENVS.values()) <= set(isolated.ENVS)
    for worker in isolated.WORKER_ENVS:
        assert (isolated.WORKERS_DIR / f"{worker}.py").exists(), worker


def test_find_uv_looks_beyond_path(monkeypatch, tmp_path):
    """A service or nohup job often lacks ~/.local/bin on PATH."""
    monkeypatch.setattr(isolated.shutil, "which", lambda name: None)
    monkeypatch.setattr(isolated.Path, "home", classmethod(lambda cls: tmp_path))
    assert isolated.find_uv() is None or not isolated.find_uv().startswith(str(tmp_path))
    uv = tmp_path / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    assert isolated.find_uv() == str(uv)


def test_interrupted_clone_is_redone(monkeypatch, tmp_path):
    """A clone cut short left a src dir that later builds trusted forever."""
    monkeypatch.setenv("LOCALKIN_HOME", str(tmp_path))
    monkeypatch.setattr(isolated, "find_uv", lambda: "/usr/bin/uv")
    root = tmp_path / "envs" / "gitdemo"
    (root / "src").mkdir(parents=True)          # what an interrupted clone leaves
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        if cmd[0] == "git":
            dest = isolated.Path(cmd[-1])
            dest.mkdir(parents=True)
            (dest / "requirements.txt").write_text("somepkg==1\n# comment\ndeepspeed==1\n")
        elif cmd[1] == "venv":
            py = isolated._python(root)
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")

    monkeypatch.setattr(isolated, "_run", fake_run)
    monkeypatch.setitem(isolated.ENVS, "gitdemo", {
        "python": "3.11", "git": "https://example.invalid/repo", "requirements": "requirements.txt",
        "exclude": ["deepspeed"], "packages": ["soundfile"]})
    isolated.ensure_env("gitdemo")
    assert any(c[0] == "git" for c in calls)
    assert (root / "src" / "requirements.txt").exists() and not (root / "src.partial").exists()
    pip = next(c for c in calls if c[1] == "pip")
    assert "somepkg==1" in pip and "soundfile" in pip and not any("deepspeed" in a for a in pip)
