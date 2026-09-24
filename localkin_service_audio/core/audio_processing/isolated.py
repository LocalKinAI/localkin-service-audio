"""
Isolated environments for models whose packages can't share one.

qwen-tts pins transformers 4.57, chatterbox-tts pins torch 2.6 and numpy<2,
IndexTTS pins torch 2.8, CosyVoice has no package at all and FireRedASR2
needs Python 3.11 — no single environment satisfies them, and the
transformers-native ASR families need transformers 5.x while the default
install stays on 4.x. So each family runs in its own virtualenv, built with
uv on first use under ~/.localkin-service-audio/envs/<env>, as a worker
process speaking the JSON-lines protocol in workers/_protocol.py.

For the user it's still one model name: the first load builds the
environment (minutes, once), later loads reuse it. Set
LOCALKIN_AUTO_INSTALL=0 to refuse building and get the command instead.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..config.settings import _default_home
from ..types import AudioResult, ModelConfig, TranscriptionResult
from .shared import (
    apply_speed, audio_duration, default_reference, detect_text_language, pick_voice,
    segments_from, to_iso, voice_infos,
)
from .stt.base import STTStrategy
from .tts.base import TTSStrategy

WORKERS_DIR = Path(__file__).resolve().parents[2] / "workers"

_TORCH = ["torch", "torchaudio", "soundfile", "librosa", "huggingface_hub"]

# env name -> how to build it. "packages" go to `uv pip install`; "git" is
# cloned (with submodules) and its requirements file installed, minus lines
# matching "exclude"; "env" is added to the worker's environment.
ENVS: Dict[str, Dict[str, Any]] = {
    "transformers": {
        "python": "3.11",
        "packages": _TORCH + ["transformers>=5.17", "accelerate", "diffusers",
                              "mistral-common[audio]", "sentencepiece"],
    },
    "qwen3_tts": {"python": "3.11", "packages": ["qwen-tts==0.1.1", "soundfile"]},
    "voxcpm": {"python": "3.11", "packages": ["voxcpm>=2.0.3", "soundfile"]},
    "omnivoice": {"python": "3.11", "packages": _TORCH + ["omnivoice>=0.2.1"]},
    "chatterbox": {"python": "3.11", "packages": ["chatterbox-tts==0.1.7", "soundfile"]},
    "indextts": {
        "python": "3.11",
        "packages": ["indextts @ git+https://github.com/index-tts/index-tts", "soundfile", "huggingface_hub"],
    },
    "cosyvoice": {
        # Not 3.10: scipy's last 3.10 wheel (1.15.3) won't load on macOS 27.
        "python": "3.11",
        "git": "https://github.com/FunAudioLLM/CosyVoice",
        "requirements": "requirements.txt",
        # GPU-server extras and the web demo; not needed for inference.
        "exclude": ["deepspeed", "tensorrt", "gradio", "fastapi", "uvicorn", "grpcio", "--extra-index-url"],
        # CosyVoice imports pkg_resources, which setuptools 81 removed.
        "packages": ["soundfile", "huggingface_hub", "setuptools<81"],
        "env": {"LOCALKIN_COSYVOICE_REPO": "{src}"},
    },
    "fireredasr2": {
        "python": "3.11",
        "packages": ["fireredasr2s @ git+https://github.com/FireRedTeam/FireRedASR2S",
                     "soundfile", "librosa", "huggingface_hub"],
    },
}

# worker script -> the environment it runs in
WORKER_ENVS = {
    "transformers_asr": "transformers",
    "transformers_tts": "transformers",
    "qwen3_tts": "qwen3_tts",
    "voxcpm": "voxcpm",
    "omnivoice": "omnivoice",
    "chatterbox": "chatterbox",
    "indextts": "indextts",
    "cosyvoice": "cosyvoice",
    "fireredasr2": "fireredasr2",
}

_build_lock = threading.Lock()


def env_root(name: str) -> Path:
    return _default_home() / "envs" / name


def _python(root: Path) -> Path:
    return root / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_uv() -> Optional[str]:
    """uv on PATH, or where its installer and package managers put it.

    Services, cron jobs and `nohup` runs often have a bare PATH without
    ~/.local/bin, where the official installer puts uv.
    """
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path(sys.executable).parent / "uv",
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _spec_hash(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def ensure_env(name: str) -> Dict[str, str]:
    """Build environment ``name`` if missing or stale; return worker env vars.

    Rebuilt when its ENVS entry changes (hash stamped in the env dir).
    """
    spec = ENVS[name]
    root = env_root(name)
    stamp = root / "localkin-env.json"
    extra_env = {k: v.format(src=root / "src") for k, v in spec.get("env", {}).items()}

    with _build_lock:
        if stamp.exists() and json.loads(stamp.read_text()).get("hash") == _spec_hash(spec) \
                and _python(root).exists():
            return extra_env

        uv = find_uv()
        if not uv:
            raise RuntimeError(
                f"Model environment '{name}' needs uv to build. Install it: "
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            )
        if os.environ.get("LOCALKIN_AUTO_INSTALL", "1") == "0":
            raise RuntimeError(
                f"Model environment '{name}' isn't built and LOCALKIN_AUTO_INSTALL=0. "
                f"Build it with: kin audio pull <model>  (or unset LOCALKIN_AUTO_INSTALL)"
            )

        print(f"Building the '{name}' model environment in {root} "
              f"(one time; can take several minutes)...", file=sys.stderr)
        root.mkdir(parents=True, exist_ok=True)
        _run([uv, "venv", "--allow-existing", "--python", spec["python"], str(root / "venv")])

        requirements = list(spec.get("packages", []))
        if spec.get("git"):
            src = root / "src"
            if not (src / spec["requirements"]).exists():
                # Clone beside it and rename, so an interrupted clone leaves
                # nothing that looks finished and the next build retries.
                partial = root / "src.partial"
                shutil.rmtree(partial, ignore_errors=True)
                shutil.rmtree(src, ignore_errors=True)
                _run(["git", "clone", "--depth", "1", "--recursive", spec["git"], str(partial)])
                partial.rename(src)
            lines = (src / spec["requirements"]).read_text().splitlines()
            requirements += [
                ln.strip() for ln in lines
                if ln.strip() and not ln.startswith("#")
                and not any(x in ln for x in spec.get("exclude", []))
            ]
        _run([uv, "pip", "install", "--python", str(_python(root)), *requirements])
        stamp.write_text(json.dumps({"hash": _spec_hash(spec), "built": time.time()}))
        return extra_env


def _run(cmd):
    print("  $ " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


class Worker:
    """One model process in its environment; requests are serialised."""

    def __init__(self, script: str):
        env_name = WORKER_ENVS[script]
        extra_env = ensure_env(env_name)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
        env.update(extra_env, PYTHONPATH=str(WORKERS_DIR), PYTHONUNBUFFERED="1")
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [str(_python(env_root(env_name))), str(WORKERS_DIR / f"{script}.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", env=env, cwd=str(WORKERS_DIR),
        )
        self._read()  # ready line

    def call(self, op: str, **kwargs) -> Dict[str, Any]:
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f"model worker exited ({self.proc.returncode}); see server log")
            self.proc.stdin.write(json.dumps({"op": op, **kwargs}) + "\n")
            self.proc.stdin.flush()
            reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "worker error"))
        return reply

    def _read(self) -> Dict[str, Any]:
        line = self.proc.stdout.readline()
        if not line:
            code = self.proc.wait()
            raise RuntimeError(f"model worker exited ({code}); see server log for its traceback")
        return json.loads(line)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.call("shutdown")
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


def _start(strategy, model_config: ModelConfig) -> bool:
    params = model_config.parameters or {}
    try:
        strategy.worker = Worker(params["worker"])
        info = strategy.worker.call(
            "load", repo=model_config.local_path or model_config.repo_id,
            params={k: v for k, v in params.items() if k != "worker"},
        )
        strategy.model = strategy.worker
        strategy.model_config = model_config
        strategy.device = info.get("device", "isolated")
        strategy._is_loaded = True
        return True
    except Exception as e:
        strategy.load_error = str(e)
        print(f"Failed to load {model_config.name}: {e}", file=sys.stderr)
        if getattr(strategy, "worker", None):
            strategy.worker.close()
        return False


class IsolatedTTSStrategy(TTSStrategy):
    """TTS in an isolated worker. Registry ``parameters``: ``worker`` (script
    name), ``needs_reference`` (fall back to the bundled voice), ``voices``,
    plus anything the worker's load() reads."""

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        return _start(self, model_config)

    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs) -> AudioResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        import soundfile as sf

        start_time = time.time()
        params = self.model_config.parameters or {}
        language = kwargs.pop("language", None) or detect_text_language(text)
        default_voice = (params.get("defaults") or {}).get("voice")
        voice = pick_voice(voice, params.get("voices"), params.get("voice_aliases"), default_voice)
        if not kwargs.get("ref_audio") and params.get("needs_reference"):
            kwargs["ref_audio"], kwargs["ref_text"] = default_reference(language)
        if isinstance(kwargs.get("ref_audio"), np.ndarray):
            raise ValueError("pass ref_audio as a file path for this model")

        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            reply = self.worker.call("synthesize", text=text, out=out, voice=voice, language=language,
                                     **kwargs)
            audio, sample_rate = sf.read(reply["path"], dtype="float32")
            audio = apply_speed(audio, speed, int(sample_rate))
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")
        finally:
            if os.path.exists(out):
                os.unlink(out)
        if audio.size == 0:
            raise RuntimeError(f"{self.model_config.name} produced no audio ({len(text)} chars)")
        return AudioResult(audio=audio, sample_rate=int(sample_rate), model=self.model_config.name,
                           voice=voice, duration=len(audio) / sample_rate,
                           processing_time=time.time() - start_time)

    def clone_voice(self, reference_audio: Union[str, np.ndarray], text: str,
                    reference_text: Optional[str] = None, **kwargs) -> AudioResult:
        return self.synthesize(text, ref_audio=reference_audio, ref_text=reference_text, **kwargs)

    def list_voices(self):
        return voice_infos(self.model_config.parameters or {}, (self.model_config.languages or ["en"])[0])

    def unload(self) -> None:
        if getattr(self, "worker", None):
            self.worker.close()
        super().unload()


class IsolatedSTTStrategy(STTStrategy):
    """STT in an isolated worker. Registry ``parameters``: ``worker``."""

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        return _start(self, model_config)

    def transcribe(self, audio: Union[np.ndarray, str], language: Optional[str] = None,
                   **kwargs) -> TranscriptionResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        start_time = time.time()
        temp_path = None
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(temp_path, audio, 16000)
            audio = temp_path
        try:
            reply = self.worker.call("transcribe", audio=audio, language=language, **kwargs)
            duration = audio_duration(audio)
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        finally:
            if temp_path:
                os.unlink(temp_path)
        return TranscriptionResult(
            text=(reply.get("text") or "").strip(),
            language=to_iso(reply.get("language")) or language,
            duration=duration,
            segments=segments_from(reply),
            model=self.model_config.name,
            engine="isolated",
            processing_time=time.time() - start_time,
        )

    def unload(self) -> None:
        if getattr(self, "worker", None):
            self.worker.close()
        super().unload()
