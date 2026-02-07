#!/usr/bin/env bash
# Synthesize speech from text using LocalKin Audio
# Usage: tts.sh "<text>" [extra kin options...]
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: tts.sh \"<text>\" [--voice VOICE] [--output FILE] [--speed SPEED]"
  exit 1
fi

TEXT="$1"
shift

exec kin audio tts "$TEXT" "$@"
