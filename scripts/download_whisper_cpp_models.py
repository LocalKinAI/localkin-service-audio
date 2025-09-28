#!/usr/bin/env python3
"""
Download script for whisper.cpp GGML models.

This script downloads the GGML format models used by whisper.cpp
from the official Hugging Face repository.
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path

# Model information
MODELS = {
    "tiny": {
        "filename": "ggml-tiny.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "size_mb": 39,
        "description": "Tiny model - fastest, lowest accuracy"
    },
    "base": {
        "filename": "ggml-base.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "size_mb": 74,
        "description": "Base model - good balance of speed and accuracy"
    },
    "small": {
        "filename": "ggml-small.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "size_mb": 244,
        "description": "Small model - higher accuracy, slower"
    },
    "medium": {
        "filename": "ggml-medium.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "size_mb": 769,
        "description": "Medium model - very high accuracy"
    },
    "large": {
        "filename": "ggml-large.bin",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
        "size_mb": 1550,
        "description": "Large model - highest accuracy"
    }
}

def download_file(url: str, dest_path: str, show_progress: bool = True) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"📥 Downloading from: {url}")
        print(f"💾 Saving to: {dest_path}")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Download with progress
        def progress_hook(count, block_size, total_size):
            if show_progress and total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r📊 Progress: {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, progress_hook)
        if show_progress:
            sys.stdout.write("\n")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download whisper.cpp GGML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_whisper_cpp_models.py tiny base
  python download_whisper_cpp_models.py --all --output-dir ./models
  python download_whisper_cpp_models.py small --output-dir ~/.cache/whisper-cpp
        """
    )

    parser.add_argument(
        "models",
        nargs="*",
        choices=list(MODELS.keys()),
        help="Model(s) to download"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="~/.cache/whisper-cpp",
        help="Output directory for downloaded models (default: ~/.cache/whisper-cpp)"
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force download even if file already exists"
    )

    args = parser.parse_args()

    # List models if requested
    if args.list:
        print("🎯 Available whisper.cpp models:")
        print("-" * 50)
        for name, info in MODELS.items():
            print(f"📦 {name:<8} | {info['size_mb']:>4}MB | {info['description']}")
        print("-" * 50)
        print("💡 Tip: Smaller models are faster but less accurate")
        return

    # Determine which models to download
    if args.all:
        models_to_download = list(MODELS.keys())
    elif args.models:
        models_to_download = args.models
    else:
        parser.print_help()
        print("\n❌ Please specify which models to download or use --all")
        return

    # Expand output directory path
    output_dir = os.path.expanduser(args.output_dir)

    print(f"🎵 LocalKin Service Audio - whisper.cpp Model Downloader")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📦 Models to download: {', '.join(models_to_download)}")
    print()

    # Download each model
    success_count = 0
    total_size = 0

    for model_name in models_to_download:
        model_info = MODELS[model_name]
        filename = model_info["filename"]
        url = model_info["url"]
        size_mb = model_info["size_mb"]

        dest_path = os.path.join(output_dir, filename)

        # Check if file already exists
        if os.path.exists(dest_path) and not args.force:
            print(f"✅ {model_name} already exists: {dest_path}")
            success_count += 1
            total_size += size_mb
            continue

        print(f"🔄 Downloading {model_name} model ({size_mb}MB)...")

        if download_file(url, dest_path):
            print(f"✅ {model_name} downloaded successfully!")
            success_count += 1
            total_size += size_mb
        else:
            print(f"❌ Failed to download {model_name}")

        print()

    # Summary
    print("=" * 60)
    if success_count == len(models_to_download):
        print("🎉 All downloads completed successfully!")
    else:
        print(f"⚠️  {success_count}/{len(models_to_download)} downloads completed")

    print(f"📊 Total size: {total_size}MB")
    print(f"📁 Models saved to: {output_dir}")

    # Instructions
    if success_count > 0:
        print("\n💡 Usage with LocalKin Service Audio:")
        print("  kin audio transcribe audio.wav --engine whisper-cpp --model_size base")
        print("  kin audio listen --engine whisper-cpp --model_size tiny")
        print("  kin audio run whisper-cpp-tiny --port 8000")

if __name__ == "__main__":
    main()
