"""
Paraformer STT Strategy - Alibaba FunASR.

Paraformer is a non-autoregressive ASR model optimized for Chinese
with integrated VAD and punctuation restoration.
"""
from typing import Optional, Union, List
import numpy as np

from .base import STTStrategy
from ...types import TranscriptionResult, ModelConfig, Segment


class ParaformerStrategy(STTStrategy):
    """
    Speech-to-Text using Alibaba's Paraformer model.

    Features:
    - Non-autoregressive (parallel decoding, faster)
    - Optimized for Chinese
    - Integrated VAD (Voice Activity Detection)
    - Punctuation restoration
    - Streaming support

    Requirements:
        pip install funasr modelscope
    """

    AVAILABLE_MODELS = {
        "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "paraformer-en": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "paraformer-multi": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch",
    }

    # VAD and punctuation models
    VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    PUNC_MODEL = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"

    def __init__(self):
        super().__init__()
        self.vad_model = None
        self.punc_model = None
        self._use_vad = True
        self._use_punc = True

    def load(
        self,
        model_config: ModelConfig,
        device: str = "auto",
        use_vad: bool = True,
        use_punc: bool = True
    ) -> bool:
        """
        Load Paraformer model with optional VAD and punctuation.

        Args:
            model_config: Model configuration
            device: Device to use
            use_vad: Load VAD model for voice activity detection
            use_punc: Load punctuation model for Chinese
        """
        try:
            from funasr import AutoModel

            self.device = self._detect_device(device)
            # FunASR doesn't support MPS
            actual_device = self.device if self.device != "mps" else "cpu"

            model_size = model_config.model_size or "paraformer-zh"

            # Get model ID
            model_id = self.AVAILABLE_MODELS.get(
                model_size,
                model_config.repo_id or self.AVAILABLE_MODELS["paraformer-zh"]
            )

            print(f"Loading Paraformer model '{model_size}'...")

            # Load main ASR model
            self.model = AutoModel(
                model=model_id,
                device=actual_device,
            )

            # Load VAD model
            self._use_vad = use_vad
            if use_vad:
                try:
                    print("Loading VAD model...")
                    self.vad_model = AutoModel(
                        model=self.VAD_MODEL,
                        device=actual_device,
                    )
                except Exception as e:
                    print(f"Warning: VAD model failed to load: {e}")
                    self.vad_model = None

            # Load punctuation model
            self._use_punc = use_punc
            if use_punc:
                try:
                    print("Loading punctuation model...")
                    self.punc_model = AutoModel(
                        model=self.PUNC_MODEL,
                        device=actual_device,
                    )
                except Exception as e:
                    print(f"Warning: Punctuation model failed to load: {e}")
                    self.punc_model = None

            self.model_config = model_config
            self._is_loaded = True
            print(f"Paraformer loaded on {self.device}")
            return True

        except ImportError:
            print("funasr not installed. Install with: pip install funasr modelscope")
            return False
        except Exception as e:
            print(f"Failed to load Paraformer: {e}")
            return False

    def transcribe(
        self,
        audio: Union[np.ndarray, str],
        language: Optional[str] = None,
        use_vad: Optional[bool] = None,
        use_punc: Optional[bool] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio using Paraformer.

        Args:
            audio: Audio data or file path
            language: Language code (ignored, Paraformer is language-specific)
            use_vad: Override VAD setting
            use_punc: Override punctuation setting
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            # Prepare audio
            if isinstance(audio, str):
                audio_input = audio
            else:
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, 16000)
                    audio_input = f.name

            # Run VAD if available and enabled
            segments_info = None
            vad_enabled = use_vad if use_vad is not None else self._use_vad
            if vad_enabled and self.vad_model:
                vad_result = self.vad_model.generate(input=audio_input)
                if vad_result:
                    segments_info = vad_result

            # Run ASR
            result = self.model.generate(
                input=audio_input,
                cache={},
            )

            # Extract text
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            text = result.get("text", "") if isinstance(result, dict) else str(result)

            # Add punctuation if available and enabled
            punc_enabled = use_punc if use_punc is not None else self._use_punc
            if punc_enabled and self.punc_model and text:
                try:
                    punc_result = self.punc_model.generate(input=text)
                    if punc_result and isinstance(punc_result, list):
                        punc_result = punc_result[0]
                    if isinstance(punc_result, dict):
                        text = punc_result.get("text", text)
                    elif isinstance(punc_result, str):
                        text = punc_result
                except Exception as e:
                    print(f"Warning: Punctuation failed: {e}")

            # Build segments if available
            segments = None
            if isinstance(result, dict) and "timestamp" in result:
                segments = []
                timestamps = result["timestamp"]
                if isinstance(timestamps, list):
                    for ts in timestamps:
                        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                            segments.append(Segment(
                                text="",  # Paraformer doesn't provide per-segment text
                                start=ts[0] / 1000.0,  # Convert ms to seconds
                                end=ts[1] / 1000.0,
                            ))

            return TranscriptionResult(
                text=text.strip(),
                language="zh",  # Paraformer is Chinese-focused
                segments=segments,
                model=self.model_config.name if self.model_config else "paraformer",
                engine="funasr",
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def unload(self) -> None:
        """Release all model resources."""
        super().unload()
        self.vad_model = None
        self.punc_model = None

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        """Paraformer variants support Chinese and English."""
        return ["zh", "en"]

    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls.AVAILABLE_MODELS.keys())
