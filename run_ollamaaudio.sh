#!/bin/bash
# LocalKin Service Audio Runner Script with Virtual Environment
# This script automatically activates the virtual environment and runs kin

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

# Run kin with all arguments
exec kin "$@"
