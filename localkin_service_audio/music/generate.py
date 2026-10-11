"""
One way into every music engine, shared by `kin audio music generate` and the
MCP server: pick the engine a model name needs, load it, and generate with
the arguments that engine takes.
"""
from typing import Any, Dict, List, Optional

from ..core.types import AudioResult, ModelConfig, ModelType
from .base import MusicEngine


def music_models() -> List[Dict[str, str]]:
    """Every music model name, without asking ComfyUI what it has installed
    (``kin audio music models`` does that)."""
    from .comfyui_strategy import MODELS as COMFY_MODELS
    from .heartmula_strategy import HeartMuLaStrategy
    from .mlx_music_strategy import mlx_music_available
    from .musicgen_strategy import MusicGenStrategy

    rows = [{"name": f"musicgen:{size}", "engine": "transformers",
             "description": f"MusicGen {size} - instrumental, 5-30 s, {req['vram_gb']} GB"}
            for size, req in MusicGenStrategy.get_model_requirements().items()]
    rows += [{"name": f"heartmula:{size}", "engine": "heartmula",
              "description": f"HeartMuLa {size} - songs with zh/en lyrics, {req['vram_gb']} GB"}
             for size, req in HeartMuLaStrategy.get_model_requirements().items()]
    for name, description in COMFY_MODELS.items():
        mlx = name == "minimax-music3" and mlx_music_available(name)
        rows.append({"name": name, "engine": "mlx-audio" if mlx else "comfyui",
                     "description": description + ("" if mlx else " (needs ComfyUI with its weights)")})
    return rows


def music_engine(model: str, backend: str = "auto", comfyui_url: Optional[str] = None) -> MusicEngine:
    """The engine that runs ``model``, not yet loaded.

    backend: "auto" (mlx-audio when it can run the model), "mlx", or
    anything else to skip mlx-audio (ComfyUI models then go to ComfyUI).
    """
    from .comfyui_strategy import ComfyUIMusicStrategy, is_comfyui_model
    from .heartmula_strategy import HeartMuLaStrategy
    from .mlx_music_strategy import MLXMusicStrategy, mlx_music_available
    from .musicgen_strategy import MusicGenStrategy

    if backend == "mlx" or (backend == "auto" and mlx_music_available(model)):
        return MLXMusicStrategy()
    if is_comfyui_model(model):
        return ComfyUIMusicStrategy(comfyui_url)
    if model.startswith("heart"):
        return HeartMuLaStrategy()
    return MusicGenStrategy()


def load_music(model: str, backend: str = "auto", comfyui_url: Optional[str] = None,
               device: str = "auto") -> MusicEngine:
    """``music_engine`` loaded; raises with the engine's reason if it can't."""
    engine = music_engine(model, backend, comfyui_url)
    if not engine.load(ModelConfig(name=model, type=ModelType.TTS, engine="music"), device=device):
        reason = getattr(engine, "load_error", None)
        raise RuntimeError(f"Failed to load {model}" + (f": {reason}" if reason else ""))
    return engine


def generate_music(engine: MusicEngine, prompt: str, *, duration: Optional[int] = None,
                   lyrics: Optional[str] = None, seed: Optional[int] = None, tags: Optional[str] = None,
                   params: Optional[Dict[str, Any]] = None, temperature: float = 1.0) -> AudioResult:
    """Generate with a loaded engine. MusicGen and HeartMuLa take no lyrics
    or seed and default to 10 s; the others use each model's own length."""
    from .comfyui_strategy import ComfyUIMusicStrategy
    from .heartmula_strategy import HeartMuLaStrategy
    from .mlx_music_strategy import MLXMusicStrategy

    if isinstance(engine, MLXMusicStrategy):
        return engine.generate(prompt, duration=duration, lyrics=lyrics, seed=seed, tags=tags)
    if isinstance(engine, ComfyUIMusicStrategy):
        return engine.generate(prompt, duration=duration, lyrics=lyrics, seed=seed, tags=tags,
                               params=params or {})
    if isinstance(engine, HeartMuLaStrategy):
        return engine.generate(prompt, duration=duration or 10, tags=tags, temperature=temperature)
    return engine.generate(prompt, duration=duration or 10, temperature=temperature)
