"""
Transcribe command - Speech-to-Text operations.
"""
import click
import os
import time
from typing import Optional

from ..utils import print_success, print_error, print_info, print_header


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--model", "-m",
    default="whisper-cpp:base",
    help="Model to use (e.g., whisper-cpp:base, faster-whisper:large-v3, sensevoice:small)"
)
@click.option(
    "--language", "-l",
    default=None,
    help="Language code (e.g., en, zh, ja). Auto-detect if not specified."
)
@click.option(
    "--output", "-o",
    default=None,
    help="Output file path. If not specified, prints to stdout."
)
@click.option(
    "--format", "-f",
    type=click.Choice(["text", "json", "srt", "vtt"]),
    default="text",
    help="Output format."
)
@click.option(
    "--timestamps/--no-timestamps",
    default=False,
    help="Include timestamps in output."
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device to use for inference."
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output."
)
def transcribe(
    audio_file: str,
    model: str,
    language: Optional[str],
    output: Optional[str],
    format: str,
    timestamps: bool,
    device: str,
    verbose: bool
):
    """
    Transcribe audio file to text.

    Examples:

        kin audio transcribe audio.wav

        kin audio transcribe audio.mp3 --model faster-whisper:large-v3

        kin audio transcribe audio.wav --language zh --output transcript.txt

        kin audio transcribe audio.wav --format srt --timestamps
    """
    if verbose:
        print_header("Transcription")
        print_info(f"Audio file: {audio_file}")
        print_info(f"Model: {model}")
        print_info(f"Device: {device}")

    try:
        from ...core import get_audio_engine

        engine = get_audio_engine()

        # Load model
        if verbose:
            print_info("Loading model...")

        if not engine._stt_strategy or engine._stt_model_name != model:
            success = engine.load_stt(model, device=device)
            if not success:
                print_error(f"Failed to load model: {model}")
                return

        # Transcribe
        if verbose:
            print_info("Transcribing...")

        start_time = time.time()
        result = engine.transcribe(audio_file, language=language)
        elapsed = time.time() - start_time

        # Format output
        if format == "text":
            output_text = result.text
        elif format == "json":
            import json
            output_data = {
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "processing_time": elapsed,
            }
            if timestamps and result.segments:
                output_data["segments"] = [
                    {"text": s.text, "start": s.start, "end": s.end}
                    for s in result.segments
                ]
            output_text = json.dumps(output_data, indent=2, ensure_ascii=False)
        elif format == "srt":
            output_text = _format_srt(result)
        elif format == "vtt":
            output_text = _format_vtt(result)

        # Output
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(output_text)
            print_success(f"Saved to {output}")
        else:
            print(output_text)

        if verbose:
            print_info(f"Transcription completed in {elapsed:.2f}s")
            if result.language:
                print_info(f"Detected language: {result.language}")

    except Exception as e:
        print_error(f"Transcription failed: {e}")
        raise click.Abort()


def _format_srt(result) -> str:
    """Format transcription result as SRT subtitle."""
    if not result.segments:
        return result.text

    lines = []
    for i, seg in enumerate(result.segments, 1):
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")

    return "\n".join(lines)


def _format_vtt(result) -> str:
    """Format transcription result as WebVTT subtitle."""
    if not result.segments:
        return f"WEBVTT\n\n{result.text}"

    lines = ["WEBVTT", ""]
    for seg in result.segments:
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")

    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
