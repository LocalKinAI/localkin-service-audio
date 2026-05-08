"""
Response formatters for transcription endpoints.

Supports JSON (default), plain text, Markdown with timestamps,
SRT subtitles, and WebVTT subtitles. The formatters are engine-
agnostic — they take a list of FormatSegment objects and produce
the requested representation.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional


# Public set of accepted response_format values.
SUPPORTED_FORMATS = frozenset({"json", "text", "markdown", "srt", "vtt"})


@dataclass
class FormatSegment:
    """Engine-agnostic segment used by the formatters."""

    start: float  # seconds
    end: float  # seconds
    text: str


def _format_timestamp(seconds: Optional[float], *, separator: str = ".") -> str:
    """Format seconds as ``HH:MM:SS<sep>mmm``.

    Use ``separator=","`` for SRT and ``separator="."`` for WebVTT.
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600 + minutes * 60)
    whole = int(secs)
    millis = int(round((secs - whole) * 1000))
    # Carry millisecond rounding overflow.
    if millis == 1000:
        millis = 0
        whole += 1
        if whole == 60:
            whole = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{millis:03d}"


def _format_short_timestamp(seconds: Optional[float]) -> str:
    """Human-friendly ``MM:SS`` (or ``HH:MM:SS`` if hours > 0)."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def to_markdown(
    text: str,
    segments: Optional[Iterable[FormatSegment]] = None,
    *,
    language: Optional[str] = None,
    duration: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    """Render transcription as Markdown.

    When ``segments`` are provided, output is structured as a list of
    timestamped segments. Otherwise the full text is shown under a
    single heading.
    """
    lines: List[str] = ["# Transcription", ""]

    meta_lines: List[str] = []
    if language:
        meta_lines.append(f"**Language:** {language}")
    if duration is not None:
        meta_lines.append(f"**Duration:** {_format_short_timestamp(duration)}")
    if model:
        meta_lines.append(f"**Model:** `{model}`")
    if meta_lines:
        # Two-trailing-space line breaks keep meta lines together in rendered MD.
        lines.append("  \n".join(meta_lines))
        lines.append("")

    seg_list = list(segments) if segments else []
    if seg_list:
        lines.append("## Segments")
        lines.append("")
        for seg in seg_list:
            stamp = (
                f"[{_format_short_timestamp(seg.start)} → "
                f"{_format_short_timestamp(seg.end)}]"
            )
            seg_text = seg.text.strip()
            if seg_text:
                lines.append(f"**{stamp}** {seg_text}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # Fallback: full transcript without timestamps.
    lines.append("## Text")
    lines.append("")
    lines.append(text.strip())
    return "\n".join(lines).rstrip() + "\n"


def to_srt(segments: Iterable[FormatSegment]) -> str:
    """Render segments as SRT subtitle format."""
    out: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg.start, separator=",")
        end = _format_timestamp(seg.end, separator=",")
        out.append(str(i))
        out.append(f"{start} --> {end}")
        out.append(seg.text.strip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def to_vtt(segments: Iterable[FormatSegment]) -> str:
    """Render segments as WebVTT subtitle format."""
    out: List[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg.start, separator=".")
        end = _format_timestamp(seg.end, separator=".")
        out.append(f"{start} --> {end}")
        out.append(seg.text.strip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"
