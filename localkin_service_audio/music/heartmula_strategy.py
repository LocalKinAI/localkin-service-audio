"""
HeartMuLa strategy - Music generation using HeartMuLa model.

HeartMuLa is a family of open-sourced music generation foundation models
that support lyrics and tags conditioning with multilingual support.

Model: HeartMuLa-oss-3B, HeartMuLa-oss-7B
Source: https://github.com/HeartMuLa/heartlib
License: Apache 2.0
"""
import numpy as np
import logging
import os
import tempfile
from typing import Optional, List
from pathlib import Path

from .base import MusicEngine
from ..core.types import AudioResult, ModelConfig

logger = logging.getLogger(__name__)


class HeartMuLaStrategy(MusicEngine):
    """
    Music generation using HeartMuLa model.

    Supports:
    - Model sizes: 3B (recommended), 7B
    - Multilingual lyrics (English, Chinese, Japanese, Korean, Spanish)
    - Tag-based style control (piano, happy, wedding, etc.)
    - Metal/CUDA support
    - Lazy loading for single GPU

    Example:
        config = ModelConfig(name="heartmula:3b", type=ModelType.TTS)
        engine = HeartMuLaStrategy()
        engine.load(config, device="auto")
        result = engine.generate(
            prompt="在月光下弹钢琴",  # Chinese lyrics
            tags="piano,romantic,slow",
            duration=30
        )
    """

    def __init__(self):
        super().__init__()
        self.model_size: Optional[str] = None
        self.generator = None
        self.version: Optional[str] = None
        self.ckpt_path: Optional[str] = None

    def load(self, model_config: ModelConfig, device: str = "auto") -> bool:
        """
        Load HeartMuLa model.

        Args:
            model_config: Configuration with model_name like "heartmula:3b"
            device: Device to use ("auto", "cpu", "cuda", "mps")

        Returns:
            True if loaded successfully
        """
        try:
            self.device = self._detect_device(device)
            self.model_config = model_config

            # Extract model size from config (e.g., "heartmula:3b" -> "3b")
            model_name = model_config.name
            if ":" in model_name:
                self.model_size = model_name.split(":")[-1].lower()
            else:
                self.model_size = "3b"  # Default to 3B

            if self.model_size not in ["3b", "7b"]:
                logger.error(f"Unsupported HeartMuLa model size: {self.model_size}")
                return False

            # Determine version string for API
            self.version = "3B" if self.model_size == "3b" else "7B"

            logger.info(f"Loading HeartMuLa model: {self.version}")
            logger.info(f"Device: {self.device}")

            # Ensure heartlib is installed (auto-install if missing)
            self._ensure_heartlib()

            try:
                from heartlib import HeartMuLaGenPipeline
            except ImportError as e:
                logger.error(
                    f"Failed to import heartlib even after install attempt. "
                    f"Please install manually:\n"
                    f"  pip install git+https://github.com/HeartMuLa/heartlib.git\n"
                    f"Error: {e}"
                )
                return False

            # Prepare checkpoint paths
            self.ckpt_path = self._prepare_checkpoints()
            if not self.ckpt_path:
                logger.error("Failed to prepare model checkpoints")
                return False

            logger.info(f"Using checkpoint path: {self.ckpt_path}")

            # Load HeartMuLa pipeline
            import torch

            # MPS workarounds for torch < 2.6
            if self.device == "mps":
                # Allow MPS to use system RAM overflow instead of hard OOM
                os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
                # torch.autocast doesn't support MPS until torch 2.6+
                self._patch_autocast_for_mps()

            device = torch.device(self.device)
            if self.device == "mps":
                dtype = torch.float16  # MPS: float16 to fit in GPU memory
            elif self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.bfloat16  # CUDA

            try:
                self.generator = HeartMuLaGenPipeline.from_pretrained(
                    pretrained_path=self.ckpt_path,
                    device=device,
                    dtype=dtype,
                    version=self.version,
                    lazy_load=(self.device != "cpu"),
                )

                # HeartCodec's detokenize uses ops unsupported on MPS.
                # Run codec on CPU instead — on Apple Silicon unified memory
                # there is no copy overhead.  Also patch _unload() which has
                # hardcoded torch.cuda calls that crash on MPS.
                if self.device == "mps":
                    import torch as _torch
                    self.generator.codec_device = _torch.device("cpu")
                    self.generator.codec_dtype = _torch.float32
                    self._patch_unload_for_mps(self.generator)

                logger.info(f"HeartMuLa {self.version} pipeline loaded")
            except Exception as e:
                logger.error(f"Failed to load HeartMuLa pipeline: {e}")
                return False

            self.model = self.generator
            self._is_loaded = True
            logger.info(f"HeartMuLa {self.version} loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load HeartMuLa model: {e}", exc_info=True)
            self._is_loaded = False
            return False

    def generate(
        self,
        prompt: str,
        duration: int = 10,
        tags: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        cfg_scale: float = 1.5,
        **kwargs
    ) -> AudioResult:
        """
        Generate music from text prompt (lyrics) and optional tags.

        Args:
            prompt: Lyrics or music description (supports Chinese, English, etc.)
            duration: Duration in seconds (default: 10, max: ~240s)
            tags: Comma-separated style tags (e.g., "piano,happy,wedding")
            temperature: Sampling temperature (default: 1.0)
            top_k: Top-k sampling parameter (default: 50)
            cfg_scale: Classifier-free guidance scale (default: 1.5)
            **kwargs: Additional arguments

        Returns:
            AudioResult with generated audio
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            import soundfile as sf

            logger.info(f"Generating music with HeartMuLa {self.version}")
            logger.info(f"Lyrics/Prompt: {prompt[:100]}...")
            if tags:
                logger.info(f"Tags: {tags}")

            max_audio_length_ms = int(duration * 1000)

            # cfg_scale > 1.0 doubles batch size (2x memory); disable on MPS
            if self.device == "mps" and cfg_scale != 1.0:
                logger.info("Disabling classifier-free guidance on MPS to save memory")
                cfg_scale = 1.0

            # HeartMuLa pipeline expects a dict with lyrics and tags strings
            inputs = {
                "lyrics": prompt,
                "tags": tags if tags else "instrumental",
            }

            # Generate to a temp wav file (pipeline writes via postprocess)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            try:
                self.generator(
                    inputs,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=top_k,
                    cfg_scale=cfg_scale,
                    save_path=temp_path,
                )

                # Read the generated wav back as numpy array
                audio, sampling_rate = sf.read(temp_path, dtype="float32")

                # Normalize if necessary
                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio = audio / max_val

                actual_duration = len(audio) / sampling_rate
                logger.info(f"Generated {actual_duration:.2f}s of audio at {sampling_rate}Hz")

                return AudioResult(
                    audio=audio,
                    sample_rate=sampling_rate,
                    model=f"heartmula-{self.model_size}",
                    duration=actual_duration,
                )
            finally:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Music generation failed: {e}", exc_info=True)
            raise

    def generate_with_metadata(
        self,
        prompt: str,
        tags: Optional[str] = None,
        duration: int = 10,
        **kwargs
    ) -> dict:
        """
        Generate music and return with metadata.

        Args:
            prompt: Lyrics
            tags: Style tags
            duration: Duration in seconds
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'audio', 'metadata'
        """
        result = self.generate(prompt, duration=duration, tags=tags, **kwargs)
        return {
            "audio": result.audio,
            "sample_rate": result.sample_rate,
            "duration": result.duration,
            "model": result.model,
            "tags": tags,
            "lyrics_preview": prompt[:50],
        }

    def unload(self) -> None:
        """Release model resources."""
        if self.generator is not None:
            try:
                self.generator._unload()
            except Exception:
                pass
            try:
                del self.generator
            except Exception:
                pass

        # Restore original torch.autocast if patched
        try:
            import torch
            if hasattr(torch, '_original_autocast'):
                torch.autocast = torch._original_autocast
                del torch._original_autocast
        except Exception:
            pass

        # Try to free GPU memory
        try:
            import gc
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

        self.generator = None
        self.model = None
        self._is_loaded = False
        logger.info("HeartMuLa model unloaded")

    def get_info(self) -> dict:
        """Return model information."""
        info = super().get_info()
        info.update({
            "model_size": self.model_size,
            "version": self.version,
            "supports_chinese": True,
            "supports_tags": True,
            "supported_languages": ["English", "Chinese", "Japanese", "Korean", "Spanish"],
        })
        return info

    @classmethod
    def get_model_sizes(cls) -> list:
        """Return list of supported model sizes."""
        return ["3b", "7b"]

    @classmethod
    def get_supported_durations(cls) -> list:
        """Return list of supported durations in seconds."""
        # HeartMuLa supports up to 240 seconds (~4 minutes)
        return [5, 10, 15, 20, 30, 60, 120, 240]

    @classmethod
    def get_model_requirements(cls) -> dict:
        """Return memory requirements for each model size."""
        return {
            "3b": {
                "vram_gb": 6,  # Can be lower with lazy loading
                "ram_gb": 8,
                "disk_gb": 6,
                "description": "Recommended for most users"
            },
            "7b": {
                "vram_gb": 16,  # Requires more VRAM
                "ram_gb": 16,
                "disk_gb": 14,
                "description": "Higher quality, requires more resources"
            },
        }

    @classmethod
    def get_available_tags(cls) -> List[str]:
        """Return list of example tags."""
        return [
            "piano", "acoustic", "electric", "synthesizer",
            "happy", "sad", "romantic", "melancholic",
            "wedding", "ambient", "upbeat", "calm",
            "orchestral", "rock", "pop", "jazz",
            "folk", "classical", "modern", "cinematic",
        ]

    @staticmethod
    def _patch_autocast_for_mps():
        """Patch torch.autocast to treat MPS as a no-op.

        torch.autocast doesn't support MPS until torch 2.6+.
        This allows HeartMuLa to run on MPS GPU without autocast,
        which is much faster than falling back to CPU.
        """
        import contextlib
        import torch

        if hasattr(torch, '_original_autocast'):
            return  # Already patched

        torch._original_autocast = torch.autocast

        class _MpsAutocast(torch._original_autocast):
            def __init__(self, device_type, **kwargs):
                if device_type == 'mps':
                    self._disabled = True
                else:
                    self._disabled = False
                    super().__init__(device_type, **kwargs)

            def __enter__(self):
                if self._disabled:
                    return self
                return super().__enter__()

            def __exit__(self, *args):
                if self._disabled:
                    return
                return super().__exit__(*args)

        torch.autocast = _MpsAutocast
        logger.info("Patched torch.autocast for MPS compatibility")

    @staticmethod
    def _patch_unload_for_mps(pipeline):
        """Replace heartlib's _unload with an MPS-safe version.

        The upstream _unload() calls torch.cuda.memory_allocated() and
        torch.cuda.empty_cache() which crash on MPS.
        """
        import gc
        import torch

        original_lazy = pipeline.lazy_load

        def _safe_unload(self_pipe=pipeline):
            if not original_lazy:
                return
            if self_pipe._mula is not None:
                logger.info("Unloading HeartMuLa from MPS")
                del self_pipe._mula
                self_pipe._mula = None
                gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            if self_pipe._codec is not None:
                logger.info("Unloading HeartCodec from CPU")
                del self_pipe._codec
                self_pipe._codec = None
                gc.collect()

        pipeline._unload = _safe_unload
        logger.info("Patched heartlib _unload for MPS compatibility")

    @staticmethod
    def _ensure_heartlib():
        """Ensure heartlib is installed (auto-install on first use)."""
        try:
            import heartlib  # noqa: F401
        except ImportError:
            print("heartlib not found. Installing from GitHub (one-time setup)...")
            import subprocess
            import sys
            subprocess.check_call(
                [
                    sys.executable, "-m", "pip", "install",
                    "git+https://github.com/HeartMuLa/heartlib.git",
                ],
            )
            print("heartlib installed successfully.")

    def _prepare_checkpoints(self) -> Optional[str]:
        """
        Prepare and download HeartMuLa checkpoints if needed.

        Returns:
            Path to checkpoint directory, or None if failed
        """
        from ..core.config.settings import _default_home

        # Check common checkpoint locations
        possible_paths = [
            _default_home() / "cache" / "heartmula",
            Path.home() / ".cache" / "heartmula",
            Path("/tmp/heartmula"),
            Path("./ckpt"),
            Path("./models/heartmula"),
        ]

        # If checkpoint path provided in config, use it
        if hasattr(self.model_config, 'checkpoint_path'):
            custom_path = Path(self.model_config.checkpoint_path)
            if custom_path.exists():
                return str(custom_path)

        # Check existing paths
        for path in possible_paths:
            if path.exists() and (path / "gen_config.json").exists():
                logger.info(f"Found existing checkpoints at {path}")
                return str(path)

        # Try to download if not found
        logger.info("Checkpoints not found locally, attempting download...")
        ckpt_path = possible_paths[0]
        ckpt_path.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download, hf_hub_download

            # Download tokenizer.json and gen_config.json from HeartMuLaGen
            config_repo = "HeartMuLa/HeartMuLaGen"
            logger.info(f"Downloading config from: {config_repo}")
            for filename in ("tokenizer.json", "gen_config.json"):
                if not (ckpt_path / filename).exists():
                    hf_hub_download(
                        repo_id=config_repo,
                        filename=filename,
                        local_dir=str(ckpt_path),
                    )

            # Download model weights
            model_id = f"HeartMuLa/HeartMuLa-oss-{self.version}"
            logger.info(f"Downloading from Hugging Face: {model_id}")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(ckpt_path / f"HeartMuLa-oss-{self.version}"),
            )

            # Download codec
            codec_model = "HeartMuLa/HeartCodec-oss-20260123"
            logger.info(f"Downloading codec: {codec_model}")
            snapshot_download(
                repo_id=codec_model,
                local_dir=str(ckpt_path / "HeartCodec-oss"),
            )

            return str(ckpt_path)

        except ImportError:
            logger.warning("huggingface_hub not installed, cannot auto-download checkpoints")
            logger.info("Please install with: pip install huggingface_hub")
            logger.info(
                f"Or manually download from:\n"
                f"  https://huggingface.co/HeartMuLa/HeartMuLaGen\n"
                f"  https://huggingface.co/HeartMuLa/HeartMuLa-oss-{self.version}\n"
                f"  https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to download checkpoints: {e}")
            return None
