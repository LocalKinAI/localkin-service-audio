"""
Helpers shared by the strategies that serve catalog models across backends
(mlx-audio, transformers, isolated workers): language handling, the bundled
reference voices, and calling ``generate`` functions with only the keyword
arguments they accept.
"""
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..types import Segment


# ISO code -> the English language name some models expect (Qwen3-ASR,
# Qwen3-TTS, OmniVoice). Models that take ISO codes get the code unchanged.
LANGUAGE_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "yue": "Cantonese", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
}


def call_supported(fn: Callable, *args, **kwargs) -> Any:
    """Call ``fn`` with only the keyword arguments its signature names.

    Nearly every mlx-audio ``generate`` also takes ``**kwargs``, but several
    forward those on to code that rejects unknown names, so accepting
    ``**kwargs`` is not taken as accepting everything. ``None`` values are
    dropped so each model keeps its own defaults.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    named = {
        name for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    usable = {k: v for k, v in kwargs.items() if v is not None and k in named}
    return fn(*args, **usable)


def detect_text_language(text: str) -> str:
    """Rough script detection — enough to pick a voice or language tag.

    Kana before Han: Japanese text contains kanji too, so testing Han first
    would route every Japanese string to Chinese.
    """
    for ch in text:
        if "぀" <= ch <= "ヿ":
            return "ja"
    for ch in text:
        if "가" <= ch <= "힯":
            return "ko"
    for ch in text:
        if "一" <= ch <= "鿿":
            return "zh"
    return "en"


def resolve_language(
    language: Optional[str],
    lang_map: Optional[Dict[str, str]],
    text: Optional[str] = None,
) -> Optional[str]:
    """Pick the value for a model's language argument.

    ``lang_map`` (from the registry entry) maps ISO codes to what the model
    wants, e.g. ``{"zh": "chinese", "en": "english"}`` or ``{"zh": "z"}``.
    Without a caller-supplied language, TTS guesses from the text's script.
    """
    if not lang_map:
        return None
    iso = (language or (detect_text_language(text) if text else "")).lower()
    if not iso or iso == "auto":
        return lang_map.get("auto")
    return lang_map.get(iso, lang_map.get("default"))


def pick_voice(
    requested: Optional[str],
    voices: Optional[List[str]],
    aliases: Optional[Dict[str, str]] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Map a requested voice onto one this model has.

    Clients written for Kokoro send its voice ids (``zf_xiaoxiao``); the
    two-letter prefix is language + gender, which ``aliases`` maps to the
    closest voice here. Unknown names (OpenAI's ``alloy``...) get the default
    rather than an error. Models without a voice list pass the name through.
    """
    if not requested:
        return default
    if not voices:
        return requested
    lookup = {v.lower(): v for v in voices}
    if requested.lower() in lookup:
        return lookup[requested.lower()]
    return (aliases or {}).get(requested.lower()[:2]) or default


def voice_infos(params: Dict[str, Any], fallback_language: str = "en") -> List["VoiceInfo"]:
    """VoiceInfo for a catalog entry's ``voices``, enriched by ``voice_info``."""
    from ..types import VoiceInfo

    info = params.get("voice_info") or {}
    return [
        VoiceInfo(
            id=v,
            name=(info.get(v) or {}).get("name", v),
            language=(info.get(v) or {}).get("language", fallback_language),
            gender=(info.get(v) or {}).get("gender"),
        )
        for v in params.get("voices") or []
    ]


def apply_speed(audio, speed: Optional[float], sample_rate: int = 24000):
    """Time-stretch speech to ``speed`` with pitch unchanged; 1.0/None is a no-op.

    Done here rather than asked of the model because most don't honour it:
    Qwen3-TTS takes a ``speed`` argument on both MLX ("not directly supported
    yet") and torch and ignores it, so a client's speed setting silently did
    nothing. WSOLA in plain numpy: it suits speech better than a phase vocoder
    (no "phasey" smear), and librosa's vocoder needs numba kernels that some
    numpy/numba pairs lack.
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not speed or abs(speed - 1.0) < 1e-3 or len(x) == 0:
        return x
    speed = float(min(max(speed, 0.25), 4.0))

    frame = max(64, int(sample_rate * 0.03))          # 30 ms
    hop_out = frame // 2
    hop_in = hop_out * speed
    tol = frame // 4                                   # alignment search, +-7.5 ms
    window = np.hanning(frame).astype(np.float32)

    out_len = int(len(x) / speed)
    pad = np.zeros(frame + tol, dtype=np.float32)
    x = np.concatenate([pad, x, pad, pad])
    y = np.zeros(out_len + 2 * frame, dtype=np.float32)
    norm = np.zeros_like(y)

    prev = None
    k = 0
    while True:
        out_pos = k * hop_out
        centre = int(round(k * hop_in)) + len(pad)
        if out_pos + frame > len(y) or centre + tol + frame > len(x):
            break
        pos = centre
        if prev is not None:
            # Pick the frame near `centre` that best continues the last one.
            natural = x[prev + hop_out: prev + hop_out + frame]
            region = x[centre - tol: centre + tol + frame]
            pos = centre - tol + int(np.argmax(np.correlate(region, natural, mode="valid")))
        y[out_pos: out_pos + frame] += x[pos: pos + frame] * window
        norm[out_pos: out_pos + frame] += window
        prev = pos
        k += 1

    y = y / np.maximum(norm, 1e-3)
    start = int(round(len(pad) / speed))
    return y[start: start + out_len].astype(np.float32)


_VOICES_DIR = Path(__file__).resolve().parents[2] / "assets" / "voices"


def default_reference(language: str) -> Tuple[str, str]:
    """Bundled reference clip and its transcript for clone-only models.

    Generated with Kokoro (Apache-2.0): zf_xiaoxiao for Chinese, af_heart
    otherwise.
    """
    tag = "zh" if language in ("zh", "yue") else "en"
    wav = _VOICES_DIR / f"default_{tag}.wav"
    return str(wav), (_VOICES_DIR / f"default_{tag}.txt").read_text(encoding="utf-8").strip()


_NAME_TO_ISO = {v.lower(): k for k, v in LANGUAGE_NAMES.items()}


def to_iso(language: Optional[str]) -> Optional[str]:
    if not language or not isinstance(language, str):
        return None
    lang = language.strip()
    return _NAME_TO_ISO.get(lang.lower(), lang.lower() if len(lang) <= 3 else lang)


def audio_duration(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def segments_from(out: Any) -> Optional[List[Segment]]:
    """Normalise the segment shapes STT backends return.

    Whisper-style dicts use start/end/text; VibeVoice uses start_time/
    end_time; Parakeet returns ``sentences`` objects instead of ``segments``.
    """
    if out is None:
        return None
    raw = _field(out, "segments", "sentences")
    if not raw or not isinstance(raw, (list, tuple)):
        return None
    segments = []
    for s in raw:
        start, end = _field(s, "start", "start_time"), _field(s, "end", "end_time")
        text = _field(s, "text", "content")
        if start is None or end is None or text is None:
            continue
        segments.append(Segment(text=str(text).strip(), start=float(start), end=float(end)))
    return segments or None
