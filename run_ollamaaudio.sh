#!/bin/bash
# OllamaAudio Runner Script with Virtual Environment
# This script automatically activates the virtual environment and runs ollamaaudio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "🔧 Run 'uv sync' first to create the virtual environment"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Run ollamaaudio with all arguments
exec ollamaaudio "$@"
