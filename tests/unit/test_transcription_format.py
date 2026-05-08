"""Tests for the transcription response formatters."""
import pytest

from localkin_service_audio.api.transcription_format import (
    FormatSegment,
    SUPPORTED_FORMATS,
    _format_short_timestamp,
    _format_timestamp,
    to_markdown,
    to_srt,
    to_vtt,
)


# --------------------------------------------------------------------------
# Timestamp formatting
# --------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "00:00:00.000"

    def test_seconds_only(self):
        assert _format_timestamp(5.5) == "00:00:05.500"

    def test_minutes(self):
        assert _format_timestamp(125.250) == "00:02:05.250"

    def test_hours(self):
        assert _format_timestamp(3 * 3600 + 12 * 60 + 7.123) == "03:12:07.123"

    def test_srt_separator(self):
        assert _format_timestamp(1.5, separator=",") == "00:00:01,500"

    def test_negative_clamped_to_zero(self):
        assert _format_timestamp(-1.0) == "00:00:00.000"

    def test_none_clamped_to_zero(self):
        assert _format_timestamp(None) == "00:00:00.000"

    def test_millisecond_rounding_overflow(self):
        # 0.9999s rounds to 1000ms; should carry to next second.
        assert _format_timestamp(0.9999) == "00:00:01.000"

    def test_short_format_under_hour(self):
        assert _format_short_timestamp(125) == "02:05"

    def test_short_format_over_hour(self):
        assert _format_short_timestamp(3725) == "01:02:05"


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

class TestMarkdown:
    def test_text_only_no_segments(self):
        md = to_markdown("Hello world.")
        assert "# Transcription" in md
        assert "Hello world." in md
        assert "## Segments" not in md
        assert "## Text" in md

    def test_with_metadata(self):
        md = to_markdown(
            "Hello.",
            language="en",
            duration=125.0,
            model="whisper:base",
        )
        assert "**Language:** en" in md
        assert "**Duration:** 02:05" in md
        assert "**Model:** `whisper:base`" in md

    def test_with_segments(self):
        segs = [
            FormatSegment(start=0.0, end=2.5, text="Hello world."),
            FormatSegment(start=2.5, end=5.0, text="This is a test."),
        ]
        md = to_markdown("Hello world. This is a test.", segs, language="en")
        assert "## Segments" in md
        assert "**[00:00 → 00:02]** Hello world." in md
        assert "**[00:02 → 00:05]** This is a test." in md

    def test_segments_take_precedence_over_text_section(self):
        segs = [FormatSegment(start=0.0, end=1.0, text="hi")]
        md = to_markdown("ignored fulltext", segs)
        assert "## Text" not in md
        assert "## Segments" in md

    def test_empty_segment_text_is_skipped(self):
        segs = [
            FormatSegment(start=0.0, end=1.0, text=""),
            FormatSegment(start=1.0, end=2.0, text="real"),
        ]
        md = to_markdown("real", segs)
        # Only the non-empty segment should render a bullet.
        assert md.count("**[") == 1


# --------------------------------------------------------------------------
# SRT
# --------------------------------------------------------------------------

class TestSRT:
    def test_basic(self):
        segs = [
            FormatSegment(start=0.0, end=2.5, text="Hello world."),
            FormatSegment(start=2.5, end=5.0, text="Second line."),
        ]
        srt = to_srt(segs)
        lines = srt.strip().split("\n")
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,000 --> 00:00:02,500"
        assert lines[2] == "Hello world."
        assert lines[3] == ""
        assert lines[4] == "2"
        assert lines[5] == "00:00:02,500 --> 00:00:05,000"
        assert lines[6] == "Second line."

    def test_uses_comma_separator(self):
        segs = [FormatSegment(start=0.5, end=1.0, text="x")]
        assert "00:00:00,500" in to_srt(segs)
        assert "00:00:00.500" not in to_srt(segs)

    def test_empty_input(self):
        assert to_srt([]).strip() == ""


# --------------------------------------------------------------------------
# VTT
# --------------------------------------------------------------------------

class TestVTT:
    def test_basic(self):
        segs = [FormatSegment(start=0.0, end=2.5, text="Hello.")]
        vtt = to_vtt(segs)
        assert vtt.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:02.500" in vtt
        assert "Hello." in vtt

    def test_uses_dot_separator(self):
        segs = [FormatSegment(start=0.5, end=1.0, text="x")]
        out = to_vtt(segs)
        assert "00:00:00.500" in out
        assert "00:00:00,500" not in out

    def test_empty_input_still_has_header(self):
        out = to_vtt([])
        assert out.startswith("WEBVTT")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

class TestSupportedFormats:
    def test_includes_all_documented(self):
        assert SUPPORTED_FORMATS == frozenset(
            {"json", "text", "markdown", "srt", "vtt"}
        )
