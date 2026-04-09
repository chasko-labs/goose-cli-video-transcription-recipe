#!/bin/bash
# transcribe-headless.sh — full pipeline: whisper + vision + merge on rocm-aibox
# called via SSH from mac mini:
#   ssh rocm-aibox "bash ~/code/projects/goose-cli-video-transcription-recipe/scripts/transcribe-headless.sh <url> [model]"
set -uo pipefail

URL="${1:?usage: transcribe-headless.sh <video-url> [model-size]}"
MODEL="${2:-medium}"
PROJECT_DIR="$HOME/code/projects/goose-cli-video-transcription-recipe"
WHISPER_IMAGE="goose-cli-video-transcription-recipe-whisper:latest"
VISION_IMAGE="goose-cli-video-transcription-recipe-vision:latest"

cd "$PROJECT_DIR"

# build images if missing
if ! docker image inspect "$WHISPER_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building whisper container (first run)"
  docker build -f docker/whisper-rocm.Dockerfile -t "$WHISPER_IMAGE" docker/
fi
if ! docker image inspect "$VISION_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building vision container (first run)"
  docker build -f docker/vision-rocm.Dockerfile -t "$VISION_IMAGE" docker/
fi

# --- stage 1: whisper (download + extract frames + transcribe) ---
echo "[pipeline] stage 1/3: whisper (model=$MODEL)"
WHISPER_OUT=$(docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v ./media/videos:/media/videos \
  -v ./media/audio:/media/audio \
  -v ./media/frames:/media/frames \
  -v ./media/transcripts:/media/transcripts \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e SCENE_THRESHOLD=0.4 \
  "$WHISPER_IMAGE" "$URL" "$MODEL" 2>&1)
echo "$WHISPER_OUT"

# extract base prefix from "[transcribe] transcript: /media/transcripts/<BASE>.*"
BASE=$(echo "$WHISPER_OUT" | grep '\[transcribe\] transcript:' | sed 's|.*transcripts/||;s|\.\*||' | tr -d ' ')
if [ -z "$BASE" ]; then
  echo "[pipeline] error: could not determine base prefix from whisper output" >&2
  exit 1
fi
echo "[pipeline] base prefix: $BASE"

# --- stage 2: vision (Instella-VL-1B frame analysis) ---
echo "[pipeline] stage 2/3: vision (Instella-VL-1B)"
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v ./media/frames:/media/frames \
  -v ./media/transcripts:/media/transcripts \
  -v goose-cli-video-transcription-recipe_hf-cache:/cache/huggingface \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VISION_MODEL=amd/Instella-VL-1B \
  "$VISION_IMAGE" /media/frames "$BASE" /media/transcripts

# --- stage 3: merge (correlate timestamps, write combined doc) ---
echo "[pipeline] stage 3/3: merge"
docker run --rm \
  -v ./media:/media \
  -v ./scripts:/scripts \
  --entrypoint python3 \
  "$WHISPER_IMAGE" /scripts/merge-outputs.py "$BASE" /media

echo ""
echo "[pipeline] done"
echo "[pipeline] combined doc: media/transcripts/${BASE}-combined.md"
echo "[pipeline] combined json: media/transcripts/${BASE}-combined.json"
