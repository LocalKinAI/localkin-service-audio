"""
Music through ComfyUI: MiniMax Music 3, ACE-Step 1.5, YuE2, Stable Audio 3.

ComfyUI runs these natively and ships a blueprint for each; this strategy
fetches the blueprint from a running ComfyUI (local or on another machine),
turns it into an API graph (comfyui_blueprint.py), fills in the prompt,
lyrics, duration and seed, queues it and downloads the result. The weights
stay wherever ComfyUI keeps them — nothing is downloaded twice.

ComfyUI's address: ``LOCALKIN_COMFYUI_URL`` or ``--comfyui-url``, default
http://localhost:8188.
"""
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

from .base import MusicEngine
from .comfyui_blueprint import BlueprintError, missing_files, subgraph_inputs, to_api_prompt
from ..core.types import AudioResult, ModelConfig

DEFAULT_URL = "http://localhost:8188"

# kin model name -> ComfyUI blueprint name
MODELS = {
    "minimax-music3": "Text to Music (MiniMax Music 3)",
    "ace-step:1.5": "Text to Audio (ACE-Step 1.5)",
    "yue2": "Text to Music (YuE2)",
    "stable-audio3": "Audio Generation (Stable Audio 3 Medium)",
    "stable-audio3:base": "Audio Generation (Stable Audio 3 Medium Base)",
}

# kin's generic arguments -> the names blueprints use for them
_ALIASES = {
    "prompt": ("caption", "prompt", "tags", "text", "positive", "description"),
    "lyrics": ("lyrics",),
    "duration": ("max_duration", "seconds", "duration", "length", "audio_length"),
    "seed": ("seed", "noise_seed"),
}


def comfyui_url(url: Optional[str] = None) -> str:
    return (url or os.environ.get("LOCALKIN_COMFYUI_URL") or DEFAULT_URL).rstrip("/")


def is_comfyui_model(name: str) -> bool:
    return name in MODELS or name.startswith("comfyui:")


def _get(url: str, timeout: float = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class ComfyUIMusicStrategy(MusicEngine):
    """Music generation by queueing a ComfyUI blueprint."""

    def __init__(self, url: Optional[str] = None, timeout: float = 1800):
        super().__init__()
        self.url = comfyui_url(url)
        self.timeout = timeout
        self.blueprint: Optional[Dict[str, Any]] = None
        self.object_info: Dict[str, Any] = {}
        self.load_error: Optional[str] = None

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        name = model_config.name
        wanted = MODELS.get(name) or name.split("comfyui:", 1)[-1]
        try:
            index = _get(f"{self.url}/api/global_subgraphs")
            by_name = {v.get("name"): k for k, v in index.items()}
            if wanted not in by_name:
                audio = sorted(n for n in by_name if n and any(w in n.lower() for w in ("music", "audio", "song")))
                raise BlueprintError(f"ComfyUI has no blueprint {wanted!r}; audio ones: {audio}")
            entry = _get(f"{self.url}/api/global_subgraphs/{by_name[wanted]}")
            data = entry.get("data")
            self.blueprint = data if isinstance(data, dict) else json.loads(data)
            self.object_info = _get(f"{self.url}/object_info", timeout=60)
            graph, _ = to_api_prompt(self.blueprint, self.object_info, {})
            missing = missing_files(graph, self.object_info)
            if missing:
                raise BlueprintError(
                    f"ComfyUI is missing model files for {wanted!r}: {', '.join(missing)}. "
                    "Install them from the template in ComfyUI (it offers the downloads)."
                )
        except BlueprintError as e:
            self.load_error = str(e)
        except OSError as e:  # URLError, timeouts
            self.load_error = (f"can't reach ComfyUI at {self.url} ({e}); is it running? "
                               "Set LOCALKIN_COMFYUI_URL or --comfyui-url")
        except ValueError as e:
            self.load_error = f"unexpected reply from ComfyUI at {self.url}: {e}"
        if self.load_error:
            print(f"Failed to load {name}: {self.load_error}")
            return False
        self.model = wanted
        self.model_config = model_config
        self.device = f"comfyui {self.url}"
        self._is_loaded = True
        return True

    def inputs(self) -> List[str]:
        return subgraph_inputs(self.blueprint) if self.blueprint else []

    def _values(self, prompt: str, duration: Optional[float], lyrics: Optional[str],
                seed: Optional[int], tags: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
        names = self.inputs()
        if tags and "tags" not in names:
            prompt = f"{prompt}, {tags}" if prompt else tags
        given = {"prompt": prompt, "lyrics": lyrics, "duration": duration, "seed": seed}
        values: Dict[str, Any] = {}
        for key, value in given.items():
            if value is None:
                continue
            target = next((n for n in _ALIASES[key] if n in names), None)
            if target:
                values[target] = value
        if tags and "tags" in names:
            values["tags"] = tags
        values.update({k: v for k, v in extra.items() if v is not None})
        return values

    def generate(
        self,
        prompt: str,
        duration: int = 60,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        lyrics: Optional[str] = None,
        seed: Optional[int] = None,
        tags: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AudioResult:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        import soundfile as sf

        start = time.time()
        values = self._values(prompt, duration, lyrics, seed, tags, params or {})
        graph, save_id = to_api_prompt(self.blueprint, self.object_info, values)
        body = json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode()
        request = urllib.request.Request(f"{self.url}/prompt", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                prompt_id = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode(errors='replace')[:500]}")

        outputs = self._wait(prompt_id)
        files = (outputs.get(save_id) or {}).get("audio") or []
        if not files:
            raise RuntimeError(f"ComfyUI finished without audio (prompt {prompt_id})")
        f = files[0]
        query = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                        "type": f.get("type", "output")})
        with urllib.request.urlopen(f"{self.url}/view?{query}", timeout=120) as r:
            audio, sample_rate = sf.read(io.BytesIO(r.read()), dtype="float32", always_2d=False)
        return AudioResult(audio=audio, sample_rate=int(sample_rate), model=self.model_config.name,
                           duration=len(audio) / sample_rate, processing_time=time.time() - start)

    def _wait(self, prompt_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            history = _get(f"{self.url}/history/{prompt_id}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    kinds = {m[0]: m[1] for m in status.get("messages", []) if m}
                    if "execution_interrupted" in kinds:
                        raise RuntimeError("interrupted in ComfyUI (someone pressed Cancel or /interrupt)")
                    detail = (kinds.get("execution_error") or {}).get("exception_message", "").strip()
                    raise RuntimeError(f"ComfyUI failed: {detail or 'see the ComfyUI log'}")
                if status.get("completed") or entry.get("outputs"):
                    return entry.get("outputs", {})
            time.sleep(2)
        raise RuntimeError(f"ComfyUI didn't finish within {self.timeout:.0f}s (prompt {prompt_id})")

    def unload(self) -> None:
        self.model = None
        self._is_loaded = False

    def get_info(self) -> Dict[str, Any]:
        return {"engine": "ComfyUIMusicStrategy", "device": self.device, "blueprint": self.model,
                "inputs": self.inputs()}


def list_audio_blueprints(url: Optional[str] = None) -> List[Dict[str, Any]]:
    """Audio blueprints on a ComfyUI, with whether their model files are there."""
    base = comfyui_url(url)
    index = _get(f"{base}/api/global_subgraphs")
    info = _get(f"{base}/object_info", timeout=60)
    reverse = {v: k for k, v in MODELS.items()}
    out = []
    for key, entry in index.items():
        name = entry.get("name") or ""
        if not any(w in name.lower() for w in ("music", "audio", "song")):
            continue
        detail = _get(f"{base}/api/global_subgraphs/{key}")
        data = detail.get("data")
        bp = data if isinstance(data, dict) else json.loads(data)
        try:
            graph, _ = to_api_prompt(bp, info, {})
            missing = missing_files(graph, info)
        except BlueprintError as e:
            missing = [f"(unsupported: {e})"]
        out.append({"model": reverse.get(name, f"comfyui:{name}"), "blueprint": name,
                    "ready": not missing, "missing": missing, "inputs": subgraph_inputs(bp)})
    return sorted(out, key=lambda m: (not m["ready"], m["model"]))
