"""
Moonshine STT Strategy - Useful Sensors ultra-fast model.

Moonshine is designed for real-time speech recognition with
extremely low latency and small model size.
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig


class MoonshineStrategy(STTStrategy):
    """
    Speech-to-Text using Useful Sensors' Moonshine model.

    Features:
    - Ultra-fast inference (5x real-time on CPU)
    - Tiny model size (~20MB for tiny)
    - Optimized for English
    - Great for real-time applications
    - Low memory usage

    Requirements:
        pip install moonshine-onnx
        # or
        pip install useful-moonshine
    """

    AVAILABLE_MODELS = {
        "moonshine-tiny": "usefulsensors/moonshine-tiny",
        "moonshine-base": "usefulsensors/moonshine-base",
    }

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load Moonshine model."""
        try:
            # Try moonshine-onnx first (faster)
            try:
                from moonshine_onnx import MoonshineOnnxModel
                self._use_onnx = True
            except ImportError:
                from moonshine import Moonshine
                self._use_onnx = False

            model_size = model_config.model_size or "base"
            if not model_size.startswith("moonshine-"):
                model_size = f"moonshine-{model_size}"

            print(f"Loading Moonshine model '{model_size}'...")

            if self._use_onnx:
                # ONNX version
                model_name = model_size.replace("moonshine-", "")
                self.model = MoonshineOnnxModel(model_name=model_name)
            else:
                # PyTorch version
                self.model = Moonshine(model_size.replace("moonshine-", ""))

            self.device = "cpu"  # Moonshine is CPU-optimized
            self.model_config = model_config
            self._is_loaded = True
            print(f"Moonshine loaded ({'ONNX' if self._use_onnx else 'PyTorch'})")
            return True

        except ImportError:
            print("Moonshine not installed. Install with: pip install moonshine-onnx")
            return False
        except Exception as e:
            print(f"Failed to load Moonshine: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using Moonshine."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Load audio
            audio_array = self._load_audio(audio)

            # Ensure correct sample rate (Moonshine expects 16kHz)
            # The base class _load_audio already handles this

            # Run transcription
            if self._use_onnx:
                result = self.model.generate(audio_array)
            else:
                result = self.model.transcribe(audio_array)

            # Handle result format
            if isinstance(result, list):
                text = " ".join(result)
            elif isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)

            return TranscriptionResult(
                text=text.strip(),
                language="en",  # Moonshine is English-only
                model=self.model_config.name if self.model_config else "moonshine",
                engine="moonshine",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Moonshine currently supports English only."""
        return ["en"]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return ["moonshine-tiny", "moonshine-base"]
