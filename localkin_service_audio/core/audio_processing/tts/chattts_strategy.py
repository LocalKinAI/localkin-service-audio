"""
ChatTTS Strategy - Conversational TTS for dialogue applications.

ChatTTS is optimized for conversational audio with:
- Natural speech patterns
- Emotion tokens ([laugh], [break])
- Chinese and English support
"""
from typing import Optional, List
import numpy as np

from .base import TTSStrategy
from ...types import AudioResult, VoiceInfo, ModelConfig


class ChatTTSStrategy(TTSStrategy):
    """
    Text-to-Speech using ChatTTS.

    Features:
    - Conversational speech style
    - Emotion tokens: [laugh], [break], [oral_*], [speed_*]
    - Multi-speaker support
    - Fine-grained prosody control
    - Chinese and English support

    Requirements:
        pip install ChatTTS
    """

    # Emotion/control tokens
    TOKENS = {
        "laugh": "[laugh]",
        "break": "[break]",
        "oral_0": "[oral_0]",  # Less filler words
        "oral_9": "[oral_9]",  # More filler words
        "speed_1": "[speed_1]",  # Slower
        "speed_9": "[speed_9]",  # Faster
    }

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """Load ChatTTS model."""
        try:
            import ChatTTS

            self.device = self._detect_device(device)

            print("Loading ChatTTS model...")

            self.model = ChatTTS.Chat()
            self.model.load(
                compile=False,  # Set True for faster inference after warmup
            )

            self.model_config = model_config
            self._is_loaded = True
            self._speaker_embedding = None

            print(f"ChatTTS loaded")
            return True

        except ImportError:
            print("ChatTTS not installed. Install with: pip install ChatTTS")
            return False
        except Exception as e:
            print(f"Failed to load ChatTTS: {e}")
            return False

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        **kwargs
    ) -> AudioResult:
        """
        Synthesize conversational speech.

        Args:
            text: Text to synthesize (can include emotion tokens)
            voice: Speaker embedding (hex string) or None for random
            speed: Speech speed (use [speed_*] tokens for finer control)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import time
        start_time = time.time()

        try:
            import ChatTTS

            # Parse speaker embedding
            spk_emb = None
            if voice:
                try:
                    import torch
                    # Voice can be a hex string or saved embedding
                    if isinstance(voice, str) and len(voice) > 20:
                        spk_emb = torch.tensor(
                            [float(x) for x in bytes.fromhex(voice)]
                        )
                except:
                    pass

            # Use saved speaker if available
            if spk_emb is None and self._speaker_embedding is not None:
                spk_emb = self._speaker_embedding

            # Build inference params
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=spk_emb,
                temperature=temperature,
                top_P=top_p,
                top_K=top_k,
            )

            # Synthesize
            wavs = self.model.infer(
                text,
                params_infer_code=params_infer_code,
                skip_refine_text=False,
                use_decoder=True,
            )

            if wavs and len(wavs) > 0:
                audio = wavs[0]
                if hasattr(audio, 'numpy'):
                    audio = audio.numpy()
                audio = audio.flatten().astype(np.float32)
            else:
                audio = np.array([], dtype=np.float32)

            sample_rate = 24000  # ChatTTS default

            return AudioResult(
                audio=audio,
                sample_rate=sample_rate,
                model=self.model_config.name if self.model_config else "chattts",
                voice=voice or "random",
                duration=len(audio) / sample_rate if len(audio) > 0 else 0,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def synthesize_with_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        **kwargs
    ) -> AudioResult:
        """
        Synthesize with emotion.

        Args:
            text: Text to synthesize
            emotion: Emotion type (happy, sad, laugh, etc.)
        """
        # Add emotion tokens
        emotion_mapping = {
            "happy": "[oral_2][laugh]",
            "sad": "[oral_7][speed_3]",
            "excited": "[oral_0][speed_7]",
            "calm": "[oral_9][speed_3]",
            "laugh": "[laugh]",
        }

        prefix = emotion_mapping.get(emotion.lower(), "")
        modified_text = f"{prefix}{text}"

        return self.synthesize(modified_text, **kwargs)

    def generate_speaker(self) -> str:
        """
        Generate a random speaker embedding.

        Returns:
            Hex string that can be used as voice parameter
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        try:
            spk_emb = self.model.sample_random_speaker()
            self._speaker_embedding = spk_emb

            # Convert to storable format
            if hasattr(spk_emb, 'numpy'):
                data = spk_emb.numpy().tobytes().hex()
            else:
                data = spk_emb.tobytes().hex()

            return data

        except Exception as e:
            raise RuntimeError(f"Failed to generate speaker: {e}")

    def list_voices(self) -> List[VoiceInfo]:
        """ChatTTS uses random speakers, so no predefined voices."""
        return [
            VoiceInfo(
                id="random",
                name="Random Speaker",
                language="zh",
                description="Randomly generated speaker",
            ),
        ]

    @classmethod
    def get_supported_languages(cls) -> List[str]:
        return ["zh", "en"]

    @classmethod
    def supports_emotion(cls) -> bool:
        return True

    @classmethod
    def supports_streaming(cls) -> bool:
        return False  # ChatTTS doesn't support streaming yet
