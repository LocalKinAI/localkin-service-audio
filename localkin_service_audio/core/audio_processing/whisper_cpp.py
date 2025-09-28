import os
import subprocess
import json
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

class WhisperCppSTT:
    """
    Integration with whisper.cpp for speech-to-text transcription.

    whisper.cpp is a C/C++ port of OpenAI's Whisper model that provides
    high-performance, local inference without Python dependencies.
    """

    def __init__(self, model_path: str = None, executable_path: str = None):
        """
        Initialize whisper.cpp STT engine.

        Args:
            model_path: Path to the GGML model file
            executable_path: Path to whisper-cli executable
        """
        self.model_path = model_path
        self.executable_path = executable_path or self._find_executable()

        if not self.executable_path:
            raise FileNotFoundError(
                "whisper-cli executable not found. Please install whisper.cpp and ensure "
                "whisper-cli is in your PATH, or provide the executable_path parameter."
            )

    def _find_executable(self) -> Optional[str]:
        """Find whisper-cli executable in common locations."""
        # Check PATH first
        try:
            result = subprocess.run(
                ["which", "whisper-cli"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        # Check common installation locations
        common_paths = [
            "/usr/local/bin/whisper-cli",
            "/usr/bin/whisper-cli",
            "/opt/whisper.cpp/whisper-cli",
            "./whisper.cpp/build/bin/whisper-cli",
            "../whisper.cpp/build/bin/whisper-cli",
        ]

        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        return None

    def transcribe(
        self,
        audio_file: str,
        model_path: str = None,
        language: str = None,
        output_format: str = "txt",
        verbose: bool = False,
        **kwargs
    ) -> str:
        """
        Transcribe audio using whisper.cpp.

        Args:
            audio_file: Path to audio file
            model_path: Path to GGML model (overrides instance model_path)
            language: Language code (e.g., 'en', 'es', 'fr')
            output_format: Output format ('txt', 'json', 'srt', 'vtt')
            verbose: Enable verbose output
            **kwargs: Additional whisper-cli arguments

        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        model = model_path or self.model_path
        if not model:
            raise ValueError("No model path provided. Set model_path in constructor or pass model_path parameter.")

        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file not found: {model}")

        # Build command arguments
        cmd = [
            self.executable_path,
            "-f", os.path.abspath(audio_file),  # Use absolute path
            "-m", model,
        ]

        # Add language if specified
        if language:
            cmd.extend(["-l", language])

        # Add verbose flag if requested
        if verbose:
            cmd.append("-v")

        # Suppress timestamps for clean output
        cmd.append("-nt")

        # Add any additional arguments
        for key, value in kwargs.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])

        # Create temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set output file path (without extension)
            output_file_base = os.path.join(temp_dir, "transcription")
            cmd.extend(["-of", output_file_base])

            try:
                # Run whisper-cli
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=temp_dir
                )

                # Read the transcription result
                # whisper-cli saves output to the path specified with -of flag
                if output_format == "txt":
                    output_file = f"{output_file_base}.txt"
                elif output_format == "json":
                    output_file = f"{output_file_base}.json"
                else:
                    # For other formats, try txt first as fallback
                    output_file = f"{output_file_base}.txt"

                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        if output_format == "json":
                            data = json.load(f)
                            return data.get("text", "").strip()
                        else:
                            return f.read().strip()
                else:
                    # If no output file, check stdout
                    return result.stdout.strip()

            except subprocess.CalledProcessError as e:
                error_msg = f"whisper-cli failed: {e.stderr}"
                if e.stdout:
                    error_msg += f"\nOutput: {e.stdout}"
                raise RuntimeError(error_msg)

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available whisper.cpp models.

        Returns:
            Dictionary of model information
        """
        # Common whisper.cpp model configurations
        models = {
            "whisper-cpp-tiny": {
                "name": "whisper-cpp-tiny",
                "size": "39MB",
                "description": "Tiny model - fastest, lowest accuracy",
                "filename": "ggml-tiny.bin",
                "download_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
            },
            "whisper-cpp-base": {
                "name": "whisper-cpp-base",
                "size": "74MB",
                "description": "Base model - good balance of speed and accuracy",
                "filename": "ggml-base.bin",
                "download_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
            },
            "whisper-cpp-small": {
                "name": "whisper-cpp-small",
                "size": "244MB",
                "description": "Small model - higher accuracy, slower",
                "filename": "ggml-small.bin",
                "download_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
            },
            "whisper-cpp-medium": {
                "name": "whisper-cpp-medium",
                "size": "769MB",
                "description": "Medium model - very high accuracy",
                "filename": "ggml-medium.bin",
                "download_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
            },
            "whisper-cpp-large": {
                "name": "whisper-cpp-large",
                "size": "1550MB",
                "description": "Large model - highest accuracy",
                "filename": "ggml-large.bin",
                "download_url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin"
            }
        }

        return models

    def download_model(self, model_name: str, download_dir: str = None) -> str:
        """
        Download a whisper.cpp model.

        Args:
            model_name: Name of the model to download
            download_dir: Directory to save the model (default: ~/.cache/whisper-cpp)

        Returns:
            Path to downloaded model file
        """
        if download_dir is None:
            download_dir = Path.home() / ".cache" / "whisper-cpp"
            download_dir.mkdir(parents=True, exist_ok=True)

        models = self.get_available_models()
        if model_name not in models:
            available = ", ".join(models.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

        model_info = models[model_name]
        filename = model_info["filename"]
        url = model_info["download_url"]

        model_path = os.path.join(download_dir, filename)

        # Check if model already exists
        if os.path.exists(model_path):
            print(f"Model already exists: {model_path}")
            return model_path

        print(f"Downloading {model_name} model...")
        print(f"URL: {url}")
        print(f"Destination: {model_path}")

        try:
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
            print(f"✅ Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def check_installation(self) -> Dict[str, Any]:
        """
        Check if whisper.cpp is properly installed.

        Returns:
            Dictionary with installation status
        """
        status = {
            "executable_found": False,
            "executable_path": None,
            "version": None,
            "supported_formats": [],
            "errors": []
        }

        # Check executable
        if self.executable_path and os.path.exists(self.executable_path):
            status["executable_found"] = True
            status["executable_path"] = self.executable_path

            # Try to get version
            try:
                result = subprocess.run(
                    [self.executable_path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    # Look for version in help output
                    help_text = result.stdout + result.stderr
                    if "whisper.cpp" in help_text:
                        status["version"] = "installed"
                    # Check for supported formats
                    if "--output-txt" in help_text or "-ot" in help_text:
                        status["supported_formats"].append("txt")
                    if "--output-json" in help_text or "-oj" in help_text:
                        status["supported_formats"].append("json")
                    if "--output-srt" in help_text or "-os" in help_text:
                        status["supported_formats"].append("srt")
                    if "--output-vtt" in help_text or "-ov" in help_text:
                        status["supported_formats"].append("vtt")
            except Exception as e:
                status["errors"].append(f"Failed to check executable: {e}")
        else:
            status["errors"].append("whisper-cli executable not found")

        return status


# Convenience functions for integration with existing codebase
def transcribe_with_whisper_cpp(
    audio_file: str,
    model_path: str,
    language: str = None,
    **kwargs
) -> str:
    """
    Convenience function to transcribe audio with whisper.cpp.

    Args:
        audio_file: Path to audio file
        model_path: Path to GGML model file
        language: Language code (optional)
        **kwargs: Additional arguments for whisper-cli

    Returns:
        Transcribed text
    """
    stt = WhisperCppSTT(model_path=model_path)
    return stt.transcribe(audio_file, language=language, **kwargs)


def get_whisper_cpp_engines() -> Dict[str, bool]:
    """
    Check availability of whisper.cpp engines.

    Returns:
        Dictionary with engine availability
    """
    try:
        stt = WhisperCppSTT()
        status = stt.check_installation()
        return {
            "whisper_cpp": status["executable_found"]
        }
    except:
        return {"whisper_cpp": False}


def get_whisper_cpp_models() -> list:
    """
    Get list of available whisper.cpp model names.

    Returns:
        List of model names
    """
    try:
        stt = WhisperCppSTT()
        models = stt.get_available_models()
        return list(models.keys())
    except:
        return []
