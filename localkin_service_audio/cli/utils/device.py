"""
Hardware detection and model recommendation utilities.
"""
from typing import Dict, List, Optional
import os
import platform

from ...core.types import HardwareProfile, DeviceType


def detect_hardware() -> HardwareProfile:
    """
    Detect system hardware capabilities.

    Returns:
        HardwareProfile with device type, memory, and CPU info
    """
    device = DeviceType.CPU
    vram_gb = 0.0
    gpu_name = None

    # Try to detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = DeviceType.CUDA
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = DeviceType.MPS
            # MPS uses unified memory, estimate ~70% available for GPU
            ram_gb = get_ram_gb()
            vram_gb = ram_gb * 0.7
            gpu_name = "Apple Silicon (MPS)"
    except ImportError:
        pass

    return HardwareProfile(
        device=device,
        vram_gb=vram_gb,
        ram_gb=get_ram_gb(),
        cpu_cores=os.cpu_count() or 1,
        gpu_name=gpu_name,
    )


def get_ram_gb() -> float:
    """System RAM in GB (8 when it can't be read)."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    try:
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            return int(result.stdout.strip()) / (1024**3)
        if platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
    except (OSError, ValueError):
        pass
    return 8.0


def recommend_models(hardware: Optional[HardwareProfile] = None) -> Dict[str, List[str]]:
    """
    Recommend STT/TTS models based on hardware capabilities.

    Args:
        hardware: HardwareProfile or None to auto-detect

    Returns:
        Dictionary with "stt" and "tts" lists of recommended model names
    """
    if hardware is None:
        hardware = detect_hardware()

    recommendations = {
        "stt": [],
        "tts": [],
        "reason": "",
    }

    # Determine tier based on available resources
    if hardware.device == DeviceType.CUDA:
        if hardware.vram_gb >= 10:
            # High-end GPU
            recommendations["stt"] = [
                "qwen3-asr:1.7b",
                "faster-whisper:large-v3-turbo",
                "parakeet:0.6b",
                "sensevoice:small",
            ]
            recommendations["tts"] = [
                "qwen3-tts:1.7b",
                "cosyvoice3:0.5b",
                "voxcpm2",
                "kokoro",
            ]
            recommendations["reason"] = f"High-end GPU detected ({hardware.gpu_name}, {hardware.vram_gb:.1f}GB VRAM)"
        elif hardware.vram_gb >= 6:
            # Mid-range GPU
            recommendations["stt"] = [
                "qwen3-asr:0.6b",
                "faster-whisper:large-v3-turbo",
                "sensevoice:small",
            ]
            recommendations["tts"] = [
                "qwen3-tts:0.6b",
                "cosyvoice3:0.5b",
                "kokoro",
            ]
            recommendations["reason"] = f"Mid-range GPU detected ({hardware.gpu_name}, {hardware.vram_gb:.1f}GB VRAM)"
        else:
            # Low-end GPU
            recommendations["stt"] = [
                "faster-whisper:large-v3-turbo",
                "sensevoice:small",
                "faster-whisper:small",
            ]
            recommendations["tts"] = [
                "kokoro",
                "native",
            ]
            recommendations["reason"] = f"Entry-level GPU detected ({hardware.gpu_name}, {hardware.vram_gb:.1f}GB VRAM)"

    elif hardware.device == DeviceType.MPS:
        # Apple Silicon
        # mlx-audio models run on Metal; see core/config/mlx_audio_models.py
        if hardware.ram_gb >= 32:
            recommendations["stt"] = [
                "qwen3-asr:1.7b",
                "fireredasr2:aed",
                "sensevoice:small",
                "whisper-mlx:large-v3-turbo",
            ]
            recommendations["tts"] = [
                "qwen3-tts:1.7b",
                "voxcpm2",
                "fish-speech:s2-pro",
                "kokoro",
            ]
            recommendations["reason"] = f"Apple Silicon with {hardware.ram_gb:.0f}GB unified memory"
        elif hardware.ram_gb >= 16:
            recommendations["stt"] = [
                "qwen3-asr:0.6b",
                "nemotron-asr:0.6b",
                "sensevoice:small",
                "whisper-mlx:large-v3-turbo",
            ]
            recommendations["tts"] = [
                "qwen3-tts:0.6b",
                "kokoro",
                "voxcpm2",
            ]
            recommendations["reason"] = f"Apple Silicon with {hardware.ram_gb:.0f}GB unified memory"
        else:
            recommendations["stt"] = [
                "qwen3-asr:0.6b",
                "whisper-cpp:base",
            ]
            recommendations["tts"] = [
                "kokoro",
                "native",
            ]
            recommendations["reason"] = f"Apple Silicon with limited memory ({hardware.ram_gb:.0f}GB)"

    else:
        # CPU only. SenseVoice isn't autoregressive, so it stays fast on a
        # CPU (Chinese, English, Cantonese, Japanese, Korean); Whisper
        # covers the other languages.
        if hardware.ram_gb >= 16:
            recommendations["stt"] = [
                "sensevoice:small",
                "faster-whisper:large-v3-turbo",
                "whisper-cpp:small",
            ]
            recommendations["tts"] = [
                "kokoro",
                "native",
            ]
            recommendations["reason"] = f"CPU with {hardware.ram_gb:.0f}GB RAM ({hardware.cpu_cores} cores)"
        else:
            recommendations["stt"] = [
                "sensevoice:small",
                "whisper-cpp:base",
            ]
            recommendations["tts"] = [
                "kokoro",
                "native",
            ]
            recommendations["reason"] = f"CPU with limited resources ({hardware.ram_gb:.0f}GB RAM)"

    return recommendations


def print_hardware_info(hardware: Optional[HardwareProfile] = None):
    """Print detected hardware information."""
    if hardware is None:
        hardware = detect_hardware()

    print("\n🖥️  Hardware Detection")
    print("=" * 40)
    print(f"Device: {hardware.device.value.upper()}")

    if hardware.gpu_name:
        print(f"GPU: {hardware.gpu_name}")
        print(f"VRAM: {hardware.vram_gb:.1f} GB")

    print(f"RAM: {hardware.ram_gb:.1f} GB")
    print(f"CPU Cores: {hardware.cpu_cores}")


def print_recommendations(hardware: Optional[HardwareProfile] = None):
    """Print model recommendations based on hardware."""
    if hardware is None:
        hardware = detect_hardware()

    recommendations = recommend_models(hardware)

    print("\n📊 Recommended Models")
    print("=" * 40)
    print(f"Based on: {recommendations['reason']}")

    print("\n🎤 Speech-to-Text (STT):")
    for model in recommendations["stt"]:
        print(f"  • {model}")

    print("\n🔊 Text-to-Speech (TTS):")
    for model in recommendations["tts"]:
        print(f"  • {model}")
