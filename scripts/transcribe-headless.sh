#!/bin/bash
# transcribe-headless.sh — full pipeline: media-extract → whisper → vision → merge → narrative
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

# fc-pool bridge endpoint for CPU-only media extraction
FC_POOL_URL="${FC_POOL_URL:-http://localhost:8150}"

# derive per-video slug from URL (no timestamp — slug identifies the video, not the run)
SLUG=$(echo "$URL" | sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60)
VIDEO_DIR="$DATA_DIR/$SLUG"

cd "$PROJECT_DIR"

# per-video subfolder + narratives dir
mkdir -p "$VIDEO_DIR"/{videos,audio,frames,transcripts} "$NARRATIVES_DIR"

# build whisper/vision images if missing
if ! docker image inspect "$WHISPER_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building whisper container (first run)"
  docker build -f docker/whisper-rocm.Dockerfile -t "$WHISPER_IMAGE" docker/
fi
if ! docker image inspect "$VISION_IMAGE" >/dev/null 2>&1; then
  echo "[pipeline] building vision container (first run)"
  docker build -f docker/vision-rocm.Dockerfile -t "$VISION_IMAGE" docker/
fi

echo "[pipeline] video dir: $VIDEO_DIR"

# --- stage 0: media-extract via fc-pool (CPU — yt-dlp + ffmpeg) ---
# Dispatches to a lightweight Firecracker VM; host receives files via ext4 data image.
# Falls back to in-container download if fc-pool is unreachable.
echo "[pipeline] stage 0/4: media-extract (fc-pool)"
MEDIA_EXTRACT_BODY=$(cat <<JSONEOF
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"extract_media","arguments":{"url":"${URL}","model":"${MODEL}"}}}
JSONEOF
)

FC_RESULT=$(curl -sf \
  --max-time 600 \
  -H "Content-Type: application/json" \
  -d "$MEDIA_EXTRACT_BODY" \
  "${FC_POOL_URL}/call/media-extract?data_dir=${VIDEO_DIR}" 2>&1) && FC_OK=true || FC_OK=false

if [ "$FC_OK" = true ]; then
  echo "[pipeline] media-extract via fc-pool done"
  # Parse BASE from the MCP result (JSON-RPC response contains {base: ...} in text content)
  BASE=$(echo "$FC_RESULT" | python3 -c "
import sys, json
try:
    resp = json.load(sys.stdin)
    content = resp.get('result', {}).get('content', [])
    for item in content:
        if item.get('type') == 'text':
            data = json.loads(item['text'])
            print(data.get('base', ''))
            break
except Exception as e:
    pass
" 2>/dev/null)
  if [ -z "$BASE" ]; then
    echo "[pipeline] warning: could not parse base from fc-pool response; checking video dir"
    BASE=$(ls -t "${VIDEO_DIR}/audio/"*.wav 2>/dev/null | head -1 | xargs -I{} basename {} .wav)
  fi
fi

if [ "$FC_OK" = false ] || [ -z "$BASE" ]; then
  echo "[pipeline] fc-pool unavailable or parse failed — falling back to in-container media extract"
  # Stage 0 fallback: run full transcribe.sh in whisper container (includes download+extract+transcribe)
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

  BASE=$(echo "$WHISPER_OUT" | grep '\[transcribe\] transcript:' | sed 's|.*transcripts/||;s|\.\*||' | tr -d ' ')
  if [ -z "$BASE" ]; then
    echo "[pipeline] error: could not determine base prefix from whisper output" >&2
    exit 1
  fi
  echo "[pipeline] base: $BASE"
  # whisper already ran in the container — skip stage 1
  WHISPER_DONE=true
fi

echo "[pipeline] base: $BASE"
WHISPER_DONE="${WHISPER_DONE:-false}"

# --- stage 1: whisper transcription (GPU) ---
# skipped if whisper already ran in the fallback path
if [ "$WHISPER_DONE" = false ]; then
  echo "[pipeline] stage 1/4: whisper transcription (model=$MODEL)"
  AUDIO_FILE=$(ls -t "${VIDEO_DIR}/audio/${BASE}"*.wav 2>/dev/null | head -1)
  if [ -z "$AUDIO_FILE" ]; then
    echo "[pipeline] error: no audio file found for base=$BASE" >&2
    exit 1
  fi

  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --group-add render \
    -v "$VIDEO_DIR/audio:/media/audio:ro" \
    -v "$VIDEO_DIR/transcripts:/media/transcripts" \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    --entrypoint whisper \
    "$WHISPER_IMAGE" \
    "/media/audio/${BASE}.wav" \
    --model "$MODEL" \
    --output_dir /media/transcripts \
    --output_format all \
    --verbose False \
    2>&1
fi

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
