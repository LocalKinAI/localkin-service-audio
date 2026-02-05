"""
WebSocket endpoints for real-time streaming transcription.

Provides real-time speech-to-text via WebSocket connection.
"""
import asyncio
import json
import numpy as np
from typing import Optional
from collections import deque

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

router = APIRouter()


class AudioBuffer:
    """Buffer for accumulating audio chunks."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_frames = 0
        self.silence_threshold = 0.01
        self.max_silence_frames = int(1.0 / chunk_duration)  # 1 second of silence

    def add(self, audio_bytes: bytes) -> None:
        """Add audio chunk to buffer."""
        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, audio])

        # Update speaking state
        energy = np.sqrt(np.mean(audio ** 2))
        if energy > self.silence_threshold:
            self.is_speaking = True
            self.silence_frames = 0
        else:
            self.silence_frames += 1

    def has_speech(self) -> bool:
        """Check if buffer has enough speech to transcribe."""
        return len(self.buffer) >= self.chunk_size and self.is_speaking

    def is_end_of_speech(self) -> bool:
        """Check if speaker has stopped (silence detected)."""
        return self.silence_frames >= self.max_silence_frames and len(self.buffer) > 0

    def get_chunk(self) -> np.ndarray:
        """Get a chunk of audio for transcription."""
        if len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            return chunk
        return np.array([], dtype=np.float32)

    def get_all(self) -> np.ndarray:
        """Get all buffered audio."""
        audio = self.buffer.copy()
        return audio

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_frames = 0


class StreamingTranscriber:
    """Real-time streaming transcription handler."""

    def __init__(
        self,
        model: str = "whisper-cpp:base",
        language: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.engine = None
        self.is_loaded = False

    async def load_model(self) -> bool:
        """Load the STT model."""
        if self.is_loaded:
            return True

        from ..core.audio_processing import get_audio_engine

        self.engine = get_audio_engine()
        success = await asyncio.to_thread(
            self.engine.load_stt,
            self.model,
        )
        self.is_loaded = success
        return success

    async def transcribe_chunk(self, audio: np.ndarray) -> dict:
        """Transcribe an audio chunk."""
        if not self.is_loaded:
            await self.load_model()

        result = await asyncio.to_thread(
            self.engine.transcribe,
            audio,
            language=self.language,
        )

        return {
            "type": "partial",
            "text": result.text,
            "language": result.language,
        }

    async def transcribe_final(self, audio: np.ndarray) -> dict:
        """Transcribe final audio segment."""
        if not self.is_loaded:
            await self.load_model()

        result = await asyncio.to_thread(
            self.engine.transcribe,
            audio,
            language=self.language,
        )

        return {
            "type": "final",
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "emotion": result.emotion,
        }

    async def handle_stream(self, websocket: WebSocket) -> None:
        """Handle WebSocket streaming transcription."""
        await websocket.accept()

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "model": self.model,
            "sample_rate": self.sample_rate,
        })

        buffer = AudioBuffer(sample_rate=self.sample_rate)

        try:
            # Load model
            if not await self.load_model():
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to load model: {self.model}",
                })
                return

            await websocket.send_json({
                "type": "model_loaded",
                "model": self.model,
            })

            # Process audio stream
            while True:
                try:
                    data = await websocket.receive_bytes()
                    buffer.add(data)

                    # Transcribe chunk if we have enough speech
                    if buffer.has_speech():
                        chunk = buffer.get_chunk()
                        if len(chunk) > 0:
                            result = await self.transcribe_chunk(chunk)
                            await websocket.send_json(result)

                    # End of speech - transcribe all remaining
                    if buffer.is_end_of_speech():
                        audio = buffer.get_all()
                        if len(audio) > 0:
                            result = await self.transcribe_final(audio)
                            await websocket.send_json(result)
                        buffer.clear()

                except WebSocketDisconnect:
                    break

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })


@router.websocket("/stream")
async def stream_transcribe(
    websocket: WebSocket,
    model: str = "whisper-cpp:base",
    language: Optional[str] = None,
):
    """
    WebSocket endpoint for real-time streaming transcription.

    Connect to this endpoint and send audio chunks as binary data.
    Audio should be 16-bit PCM at 16kHz mono.

    Messages received:
    - {"type": "ready"} - Server is ready
    - {"type": "model_loaded"} - Model loaded
    - {"type": "partial", "text": "..."} - Partial transcription
    - {"type": "final", "text": "..."} - Final transcription
    - {"type": "error", "message": "..."} - Error

    Example client:
        async with websockets.connect("ws://localhost:8000/api/stream") as ws:
            # Send audio chunks
            for chunk in audio_chunks:
                await ws.send(chunk)
                response = await ws.recv()
                print(json.loads(response))
    """
    transcriber = StreamingTranscriber(
        model=model,
        language=language,
    )
    await transcriber.handle_stream(websocket)


@router.websocket("/stream/tts")
async def stream_tts(
    websocket: WebSocket,
    model: str = "kokoro",
    voice: Optional[str] = None,
):
    """
    WebSocket endpoint for streaming TTS.

    Send text messages and receive audio chunks.

    Messages sent:
    - {"text": "Hello world"} - Text to synthesize

    Messages received:
    - {"type": "audio", "data": "<base64>"} - Audio chunk
    - {"type": "done"} - Synthesis complete
    - {"type": "error", "message": "..."} - Error
    """
    await websocket.accept()

    from ..core.audio_processing import get_audio_engine
    import base64

    engine = get_audio_engine()

    try:
        # Load model
        if not engine._tts_strategy or engine._tts_model_name != model:
            success = await asyncio.to_thread(engine.load_tts, model)
            if not success:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Failed to load model: {model}",
                })
                return

        await websocket.send_json({
            "type": "ready",
            "model": model,
        })

        while True:
            try:
                data = await websocket.receive_json()
                text = data.get("text", "")

                if text:
                    # Synthesize
                    result = await asyncio.to_thread(
                        engine.synthesize,
                        text,
                        voice=voice or data.get("voice"),
                    )

                    # Send audio as base64
                    audio_bytes = result.to_wav_bytes()
                    await websocket.send_json({
                        "type": "audio",
                        "data": base64.b64encode(audio_bytes).decode(),
                        "sample_rate": result.sample_rate,
                        "duration": result.duration,
                    })

                    await websocket.send_json({"type": "done"})

            except WebSocketDisconnect:
                break

    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e),
        })
