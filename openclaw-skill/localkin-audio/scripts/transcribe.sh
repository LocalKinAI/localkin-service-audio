#!/usr/bin/env bash
# Transcribe an audio file using LocalKin Audio
# Usage: transcribe.sh <audio_file> [extra kin options...]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: transcribe.sh <audio_file> [--model MODEL] [--language LANG] [--format FORMAT] [--output FILE]"
  exit 1
fi

AUDIO_FILE="$1"
shift

if [ ! -f "$AUDIO_FILE" ]; then
  echo "Error: File not found: $AUDIO_FILE"
  exit 1
fi

exec kin audio transcribe "$AUDIO_FILE" "$@"
