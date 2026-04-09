#!/bin/bash
# transcribe-headless.sh — full pipeline: whisper + vision + merge + narrative
# called via SSH from mac mini:
#   ssh rocm-aibox "bash ~/code/projects/goose-cli-video-transcription-recipe/scripts/transcribe-headless.sh <url> [model]"
set -uo pipefail

URL="${1:?usage: transcribe-headless.sh <video-url> [model-size]}"
MODEL="${2:-medium}"
PROJECT_DIR="$HOME/code/projects/goose-cli-video-transcription-recipe"
DATA_DIR="${VIDEO_TRANSCRIPTS_DIR:-$HOME/video-transcripts}"
WHISPER_IMAGE="goose-cli-video-transcription-recipe-whisper:latest"
VISION_IMAGE="goose-cli-video-transcription-recipe-vision:latest"
NARRATIVES_DIR="$DATA_DIR/narratives"

# derive per-video slug from URL (no timestamp — slug identifies the video, not the run)
SLUG=$(echo "$URL" | sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60)
VIDEO_DIR="$DATA_DIR/$SLUG"

cd "$PROJECT_DIR"

# per-video subfolder + narratives dir
mkdir -p "$VIDEO_DIR"/{videos,audio,frames,transcripts} "$NARRATIVES_DIR"

# build images if missing
if ! docker image inspect "$WHISPER_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building whisper container (first run)"
  docker build -f docker/whisper-rocm.Dockerfile -t "$WHISPER_IMAGE" docker/
fi
if ! docker image inspect "$VISION_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building vision container (first run)"
  docker build -f docker/vision-rocm.Dockerfile -t "$VISION_IMAGE" docker/
fi

echo "[pipeline] video dir: $VIDEO_DIR"

# --- stage 1: whisper (metadata + download + extract frames + transcribe) ---
echo "[pipeline] stage 1/4: whisper (model=$MODEL)"
WHISPER_OUT=$(docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v "$VIDEO_DIR/videos:/media/videos" \
  -v "$VIDEO_DIR/audio:/media/audio" \
  -v "$VIDEO_DIR/frames:/media/frames" \
  -v "$VIDEO_DIR/transcripts:/media/transcripts" \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e SCENE_THRESHOLD=0.4 \
  "$WHISPER_IMAGE" "$URL" "$MODEL" 2>&1)
echo "$WHISPER_OUT"

# BASE is now just a timestamp (slug is the folder)
BASE=$(echo "$WHISPER_OUT" | grep '\[transcribe\] transcript:' | sed 's|.*transcripts/||;s|\.\*||' | tr -d ' ')
if [ -z "$BASE" ]; then
  echo "[pipeline] error: could not determine base prefix from whisper output" >&2
  exit 1
fi
echo "[pipeline] base: $BASE"

# --- stage 2: vision (Instella-VL-1B frame analysis) ---
echo "[pipeline] stage 2/4: vision (Instella-VL-1B)"
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v "$VIDEO_DIR/frames:/media/frames" \
  -v "$VIDEO_DIR/transcripts:/media/transcripts" \
  -v goose-cli-video-transcription-recipe_hf-cache:/cache/huggingface \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VISION_MODEL=amd/Instella-VL-1B \
  "$VISION_IMAGE" /media/frames "$BASE" /media/transcripts

# --- stage 3: merge (correlate timestamps, write combined doc) ---
echo "[pipeline] stage 3/4: merge"
docker run --rm \
  -v "$VIDEO_DIR:/media" \
  -v "$PROJECT_DIR/scripts:/scripts" \
  --entrypoint python3 \
  "$WHISPER_IMAGE" /scripts/merge-outputs.py "$BASE" /media

# --- stage 4: narrative (LLM weave of transcript + visual into prose) ---
echo "[pipeline] stage 4/4: narrative"
python3 "$PROJECT_DIR/scripts/generate-narrative.py" "$VIDEO_DIR" "$NARRATIVES_DIR"

echo ""
echo "[pipeline] done — $VIDEO_DIR"
echo "[pipeline] combined:   $VIDEO_DIR/transcripts/${BASE}-combined.md"
echo "[pipeline] narrative:  $NARRATIVES_DIR/ (see above for filename)"
