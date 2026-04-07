#!/bin/bash
# transcribe-headless.sh — headless video transcription on rocm-aibox
# called via SSH from mac mini: ssh rocm-aibox "bash ~/code/projects/goose-cli-video-transcription-recipe/scripts/transcribe-headless.sh <url> [model]"
set -uo pipefail

URL="${1:?usage: transcribe-headless.sh <video-url> [model-size]}"
MODEL="${2:-medium}"
PROJECT_DIR="$HOME/code/projects/goose-cli-video-transcription-recipe"

cd "$PROJECT_DIR"

# build if image doesn't exist
if ! docker image inspect transcribe-whisper >/dev/null 2>&1; then
  echo "[transcribe] building whisper container (first run — this takes a while)"
  docker compose build whisper
fi

echo "[transcribe] starting pipeline: $URL (model: $MODEL)"
docker compose run --rm whisper "$URL" "$MODEL"
