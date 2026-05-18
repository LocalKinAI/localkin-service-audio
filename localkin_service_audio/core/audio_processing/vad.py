"""Standalone Voice Activity Detection (VAD) utilities.

This module exposes engine-agnostic VAD that can be used:

  - As a preprocessing step before transcription (skip silence).
  - Via the ``/vad`` HTTP endpoint to detect speech segments without
    transcribing.
  - As a library API for users embedding the package.

Currently supported backend:

  - **TEN-VAD** (``ten-vad`` on PyPI) — 731KB on Apple Silicon, ships a
    native macOS arm64 binary, ~0.016 RTF on M1. Faster speech↔silence
    transitions than Silero VAD (which is the one used internally by
    ``faster-whisper`` via its ``vad_filter`` flag — that one stays put;
    this module is additive).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class SpeechSegment:
    """A contiguous span of detected speech."""

    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


class TenVADBackend:
    """Thin wrapper around the ``ten-vad`` package.

    Loads the native handler lazily. Audio must be 16 kHz mono int16
    when passed as a numpy array; file paths are auto-resampled via
    librosa.

    Usage::

        vad = TenVADBackend(threshold=0.5)
        segments = vad.detect_speech("meeting.wav")
        # → [SpeechSegment(start=0.5, end=4.2), ...]
    """

    SAMPLE_RATE = 16_000
    HOP_SIZE = 256  # 16 ms @ 16 kHz — matches ten-vad's default

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 100,
    ) -> None:
        try:
            from ten_vad import TenVad  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ten-vad is not installed. Install with: pip install ten-vad"
            ) from e

        self.threshold = float(threshold)
        self.min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        self.min_silence_samples = int(min_silence_duration_ms * self.SAMPLE_RATE / 1000)
        self.pad_samples = int(speech_pad_ms * self.SAMPLE_RATE / 1000)
        self._vad = TenVad(hop_size=self.HOP_SIZE, threshold=self.threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_speech(
        self,
        audio: Union[str, np.ndarray],
    ) -> List[SpeechSegment]:
        """Detect speech segments in an audio file or numpy array.

        Args:
            audio: Path to an audio file (any format librosa can load),
                or a numpy array. Arrays must be either int16 at 16 kHz
                mono, or float32/float64 in [-1, 1] at 16 kHz mono.

        Returns:
            A list of :class:`SpeechSegment` covering all detected
            speech, with hangover / merge logic applied.
        """
        audio_i16 = self._load_audio_int16(audio)
        flags = self._run_frame_classifier(audio_i16)
        return self._flags_to_segments(flags, total_samples=len(audio_i16))

    def detect_speech_mask(
        self,
        audio: Union[str, np.ndarray],
    ) -> np.ndarray:
        """Return a per-frame boolean array (one entry per hop).

        Useful when callers want raw VAD output before merge / hangover
        post-processing.
        """
        audio_i16 = self._load_audio_int16(audio)
        return self._run_frame_classifier(audio_i16)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_audio_int16(self, audio: Union[str, np.ndarray]) -> np.ndarray:
        """Normalize any audio input to 16kHz mono int16 numpy."""
        if isinstance(audio, str):
            import librosa

            arr, _ = librosa.load(audio, sr=self.SAMPLE_RATE, mono=True)
            audio = arr

        arr = np.asarray(audio)
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(1, arr.ndim)))

        if arr.dtype == np.int16:
            return arr

        # Float path — clip and scale.
        arr_f = arr.astype(np.float32)
        arr_f = np.clip(arr_f, -1.0, 1.0)
        return (arr_f * 32767.0).astype(np.int16)

    def _run_frame_classifier(self, audio_i16: np.ndarray) -> np.ndarray:
        """Slide ten-vad over the audio at HOP_SIZE granularity."""
        n_frames = len(audio_i16) // self.HOP_SIZE
        flags = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            start = i * self.HOP_SIZE
            frame = audio_i16[start : start + self.HOP_SIZE]
            # ten-vad returns (probability, binary_flag); we use the flag.
            _, is_speech = self._vad.process(frame)
            flags[i] = bool(is_speech)
        return flags

    def _flags_to_segments(
        self,
        flags: np.ndarray,
        *,
        total_samples: int,
    ) -> List[SpeechSegment]:
        """Convert per-frame flags into merged speech segments.

        Applies three rules:
          1. Drop speech runs shorter than ``min_speech_duration_ms``.
          2. Merge speech runs separated by less than
             ``min_silence_duration_ms``.
          3. Pad each kept segment by ``speech_pad_ms`` on both sides
             (clamped to audio bounds).
        """
        if not len(flags):
            return []

        sr = self.SAMPLE_RATE
        hop = self.HOP_SIZE

        # 1) Collect raw speech runs in sample space.
        runs: List[List[int]] = []
        in_run = False
        run_start = 0
        for i, is_speech in enumerate(flags):
            if is_speech and not in_run:
                run_start = i * hop
                in_run = True
            elif not is_speech and in_run:
                runs.append([run_start, i * hop])
                in_run = False
        if in_run:
            runs.append([run_start, len(flags) * hop])

        # 2) Merge runs separated by short silence.
        merged: List[List[int]] = []
        for run in runs:
            if merged and run[0] - merged[-1][1] < self.min_silence_samples:
                merged[-1][1] = run[1]
            else:
                merged.append(run)

        # 3) Drop too-short and pad survivors.
        segments: List[SpeechSegment] = []
        for start_samp, end_samp in merged:
            if end_samp - start_samp < self.min_speech_samples:
                continue
            start_samp = max(0, start_samp - self.pad_samples)
            end_samp = min(total_samples, end_samp + self.pad_samples)
            segments.append(
                SpeechSegment(start=start_samp / sr, end=end_samp / sr)
            )
        return segments


def detect_speech(
    audio: Union[str, np.ndarray],
    *,
    backend: str = "ten-vad",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 200,
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 100,
) -> List[SpeechSegment]:
    """Top-level helper — run VAD on an audio input.

    Currently only ``backend="ten-vad"`` is supported. Future backends
    (Silero standalone, pyannote-segmentation) can be added behind this
    same function.
    """
    if backend != "ten-vad":
        raise ValueError(
            f"Unsupported VAD backend {backend!r}. Currently supported: 'ten-vad'."
        )
    vad = TenVADBackend(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
    return vad.detect_speech(audio)


SUPPORTED_VAD_BACKENDS = frozenset({"ten-vad"})
