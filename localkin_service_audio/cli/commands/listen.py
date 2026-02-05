"""
Listen command - Real-time STT/TTS loop with optional LLM integration.
"""
import click
import time
import numpy as np
from typing import Optional
from queue import Queue
from threading import Thread

from ..utils import print_success, print_error, print_info, print_header, print_warning


# Conversation memory for LLM context
_conversation_memory = []


def clear_conversation_memory():
    """Clear conversation history."""
    global _conversation_memory
    _conversation_memory = []


def add_to_conversation(role: str, content: str):
    """Add message to conversation history."""
    global _conversation_memory
    _conversation_memory.append({"role": role, "content": content})
    # Keep last 10 exchanges (20 messages)
    if len(_conversation_memory) > 20:
        _conversation_memory = _conversation_memory[-20:]


def get_llm_response(text: str, model: str = "qwen3:14b", stream: bool = False):
    """Get response from Ollama LLM."""
    try:
        import requests

        add_to_conversation("user", text)

        messages = _conversation_memory.copy()

        if stream:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=60,
            )

            full_response = ""
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        chunk = data["message"]["content"]
                        full_response += chunk
                        yield chunk

            add_to_conversation("assistant", full_response)
        else:
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=60,
            )
            data = response.json()
            content = data.get("message", {}).get("content", "")
            add_to_conversation("assistant", content)
            yield content

    except Exception as e:
        yield f"LLM error: {e}"


@click.command()
@click.option(
    "--model", "-m",
    default="whisper-cpp:base",
    help="STT model to use (e.g., whisper-cpp:base, faster-whisper:large-v3)"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda", "mps"]),
    default="auto",
    help="Device for inference"
)
@click.option(
    "--tts/--no-tts",
    default=False,
    help="Enable text-to-speech output"
)
@click.option(
    "--tts-model",
    default="native",
    help="TTS model to use (e.g., native, kokoro)"
)
@click.option(
    "--tts-voice",
    default=None,
    help="Voice for TTS (model-specific)"
)
@click.option(
    "--llm",
    type=click.Choice(["ollama"]),
    default=None,
    help="Enable LLM integration"
)
@click.option(
    "--llm-model",
    default="qwen3:14b",
    help="LLM model to use"
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream LLM responses"
)
@click.option(
    "--language", "-l",
    default=None,
    help="Language code (e.g., en, zh)"
)
@click.option(
    "--silence-threshold",
    default=0.01,
    type=float,
    help="Silence detection threshold (0.0-1.0)"
)
@click.option(
    "--silence-duration",
    default=1.5,
    type=float,
    help="Seconds of silence before processing"
)
def listen(
    model: str,
    device: str,
    tts: bool,
    tts_model: str,
    tts_voice: Optional[str],
    llm: Optional[str],
    llm_model: str,
    stream: bool,
    language: Optional[str],
    silence_threshold: float,
    silence_duration: float,
):
    """
    Real-time speech recognition with optional TTS and LLM.

    Listens to microphone input, transcribes speech, and optionally
    responds with TTS or sends to an LLM for conversation.

    Examples:

        kin audio listen

        kin audio listen --tts --tts-model kokoro

        kin audio listen --llm ollama --tts --stream

        kin audio listen --model sensevoice:small --language zh
    """
    print_header("Real-time Listening")

    try:
        import sounddevice as sd
    except ImportError:
        print_error("sounddevice not installed. Run: pip install sounddevice")
        return

    # Load STT model
    print_info(f"Loading STT model: {model}")
    try:
        from ...core import get_audio_engine
        engine = get_audio_engine()

        if not engine._stt_strategy or engine._stt_model_name != model:
            success = engine.load_stt(model, device=device)
            if not success:
                print_error(f"Failed to load STT model: {model}")
                return
        print_success(f"STT model loaded: {model}")
    except Exception as e:
        print_error(f"Failed to load STT model: {e}")
        return

    # Load TTS model if enabled
    if tts:
        print_info(f"Loading TTS model: {tts_model}")
        try:
            if not engine._tts_strategy or engine._tts_model_name != tts_model:
                success = engine.load_tts(tts_model, device=device)
                if not success:
                    print_error(f"Failed to load TTS model: {tts_model}")
                    tts = False
                else:
                    print_success(f"TTS model loaded: {tts_model}")
        except Exception as e:
            print_warning(f"TTS disabled: {e}")
            tts = False

    # Audio settings
    sample_rate = 16000
    channels = 1
    audio_queue = Queue()

    # State tracking
    speech_detected = False
    silence_counter = 0
    speech_buffer = []
    silence_chunks_threshold = int(silence_duration * 10)  # 100ms chunks

    def audio_callback(indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print_warning(f"Audio status: {status}")
        audio_queue.put(indata.copy())

    def speak_text(text: str):
        """Synthesize and play text."""
        if not tts:
            return
        try:
            result = engine.synthesize(text, voice=tts_voice)
            sd.play(result.audio, result.sample_rate)
            sd.wait()
        except Exception as e:
            print_warning(f"TTS failed: {e}")

    def process_speech(audio_data: np.ndarray):
        """Process accumulated speech audio."""
        nonlocal speech_detected, silence_counter, speech_buffer

        try:
            # Transcribe
            result = engine.transcribe(audio_data, language=language)
            text = result.text.strip()

            if not text:
                return

            print_success(f"You: {text}")

            # LLM response
            if llm:
                print_info("Thinking...")
                response_text = ""

                for chunk in get_llm_response(text, llm_model, stream):
                    if stream:
                        print(chunk, end="", flush=True)
                    response_text += chunk

                if stream:
                    print()  # Newline after streaming

                print_info(f"Assistant: {response_text}")

                if tts and response_text:
                    speak_text(response_text)
            elif tts:
                # Just echo back with TTS
                speak_text(text)

        except Exception as e:
            print_error(f"Processing error: {e}")

    # Main listening loop
    print_info("Starting audio stream...")
    print_info("Speak into your microphone. Press Ctrl+C to stop.")

    try:
        with sd.InputStream(
            callback=audio_callback,
            channels=channels,
            samplerate=sample_rate,
            blocksize=int(sample_rate * 0.1),  # 100ms blocks
        ):
            print_success("Listening...")

            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunk = chunk.flatten().astype(np.float32)

                    # Calculate RMS for speech detection
                    rms = np.sqrt(np.mean(chunk ** 2))

                    if rms > silence_threshold:
                        # Speech detected
                        if not speech_detected:
                            speech_detected = True
                            silence_counter = 0
                            speech_buffer = []
                            print_info("Speech detected...")

                        speech_buffer.append(chunk)
                        silence_counter = 0

                    elif speech_detected:
                        # Silence during speech
                        silence_counter += 1
                        speech_buffer.append(chunk)

                        if silence_counter >= silence_chunks_threshold:
                            # End of speech - process it
                            if speech_buffer:
                                audio_data = np.concatenate(speech_buffer)
                                process_speech(audio_data)

                            # Reset state
                            speech_detected = False
                            silence_counter = 0
                            speech_buffer = []
                            print_info("Ready...")

                except Exception:
                    pass  # Queue timeout is normal

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print_error(f"Listening failed: {e}")
        print_info("Possible causes:")
        print_info("  - No microphone available")
        print_info("  - Audio permissions not granted")
        print_info("  - Audio device busy")
    finally:
        if llm:
            clear_conversation_memory()
