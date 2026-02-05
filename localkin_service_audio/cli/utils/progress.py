"""
Progress display utilities - Ollama-style progress bars.
"""
from typing import Optional, Callable
import sys
import time


class OllamaStyleProgress:
    """
    Progress display that mimics Ollama's clean output style.

    Example output:
        pulling manifest
        pulling 8934d96d3f08... 100% |████████████████████| 3.8 GB
        pulling 8c17c2ebb0ea... 100% |████████████████████| 7.0 KB
        verifying sha256 digest
        writing manifest
        success
    """

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich
        self._progress = None

        if use_rich:
            try:
                from rich.progress import (
                    Progress,
                    SpinnerColumn,
                    TextColumn,
                    BarColumn,
                    DownloadColumn,
                    TransferSpeedColumn,
                    TimeRemainingColumn,
                )
                self._rich_available = True
            except ImportError:
                self._rich_available = False
                self.use_rich = False

    def status(self, message: str):
        """Display a status message."""
        print(message)

    def download(
        self,
        description: str,
        total: Optional[int] = None,
        callback: Optional[Callable[[int], None]] = None
    ):
        """
        Create a download progress context.

        Usage:
            with progress.download("Pulling model...", total=1000) as update:
                for chunk in download_chunks():
                    update(len(chunk))
        """
        if self.use_rich and self._rich_available:
            return self._rich_download(description, total)
        else:
            return self._simple_download(description, total)

    def _rich_download(self, description: str, total: Optional[int]):
        """Rich-based progress bar."""
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            DownloadColumn,
            TransferSpeedColumn,
            TimeRemainingColumn,
        )

        class RichDownloadContext:
            def __init__(self, desc: str, total_size: Optional[int]):
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                )
                self.desc = desc
                self.total = total_size
                self.task_id = None

            def __enter__(self):
                self.progress.__enter__()
                self.task_id = self.progress.add_task(self.desc, total=self.total or 0)
                return self._update

            def _update(self, advance: int):
                self.progress.update(self.task_id, advance=advance)

            def __exit__(self, *args):
                self.progress.__exit__(*args)

        return RichDownloadContext(description, total)

    def _simple_download(self, description: str, total: Optional[int]):
        """Simple text-based progress."""
        class SimpleDownloadContext:
            def __init__(self, desc: str, total_size: Optional[int]):
                self.desc = desc
                self.total = total_size or 0
                self.current = 0
                self.last_percent = -1

            def __enter__(self):
                print(f"{self.desc}...", end="", flush=True)
                return self._update

            def _update(self, advance: int):
                self.current += advance
                if self.total > 0:
                    percent = int(100 * self.current / self.total)
                    if percent != self.last_percent and percent % 10 == 0:
                        print(f" {percent}%", end="", flush=True)
                        self.last_percent = percent

            def __exit__(self, *args):
                print(" done")

        return SimpleDownloadContext(description, total)

    def spinner(self, message: str):
        """
        Create a spinner context for indeterminate operations.

        Usage:
            with progress.spinner("Processing..."):
                do_something()
        """
        if self.use_rich and self._rich_available:
            return self._rich_spinner(message)
        else:
            return self._simple_spinner(message)

    def _rich_spinner(self, message: str):
        """Rich-based spinner."""
        from rich.console import Console
        from rich.status import Status

        class RichSpinnerContext:
            def __init__(self, msg: str):
                self.console = Console()
                self.status = Status(msg, console=self.console)

            def __enter__(self):
                self.status.__enter__()
                return self

            def update(self, message: str):
                self.status.update(message)

            def __exit__(self, *args):
                self.status.__exit__(*args)

        return RichSpinnerContext(message)

    def _simple_spinner(self, message: str):
        """Simple text-based spinner."""
        class SimpleSpinnerContext:
            def __init__(self, msg: str):
                self.msg = msg

            def __enter__(self):
                print(f"{self.msg}...", end="", flush=True)
                return self

            def update(self, message: str):
                print(f"\r{message}...", end="", flush=True)

            def __exit__(self, *args):
                print(" done")

        return SimpleSpinnerContext(message)

    def transcribe_progress(self, audio_duration: float):
        """
        Progress bar for transcription.

        Shows estimated progress based on audio duration and processing time.
        """
        if self.use_rich and self._rich_available:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
        else:
            return None


# Global progress instance
progress = OllamaStyleProgress()
