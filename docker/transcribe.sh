#!/bin/bash
# transcribe.sh — pipeline entrypoint (runs inside whisper-rocm container)
# usage: transcribe.sh <video-url> [model-size]
# output: timestamped transcript + frames in /media/
# note: /media/{videos,audio,frames,transcripts} are mounted from the host
#       per-video slug subfolder is handled by the host (transcribe-headless.sh)
set -uo pipefail

URL="${1:?usage: transcribe.sh <video-url> [model-size]}"
MODEL="${2:-medium}"
SCENE_THRESHOLD="${SCENE_THRESHOLD:-0.4}"

AUDIO_DIR="/media/audio"
FRAMES_DIR="/media/frames"
TRANSCRIPT_DIR="/media/transcripts"
VIDEO_DIR="/media/videos"
mkdir -p "$AUDIO_DIR" "$FRAMES_DIR" "$TRANSCRIPT_DIR" "$VIDEO_DIR"

# BASE is timestamp only — slug is the host-side folder name
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BASE="$TIMESTAMP"

# --- metadata: fetch before download so we have title/uploader for context ---
echo "[transcribe] fetching metadata"
# 2>/dev/null: suppress warnings that would corrupt the JSON file
yt-dlp --dump-json "$URL" 2>/dev/null | grep "^{" > "${TRANSCRIPT_DIR}/metadata.json" \
  || echo "[transcribe] warning: metadata fetch failed (non-fatal)"

# download full video — needed for both audio and frame extraction
echo "[transcribe] downloading video from $URL"
yt-dlp --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
  --merge-output-format mp4 \
  -o "${VIDEO_DIR}/${BASE}.%(ext)s" \
  "$URL" 2>&1

VIDEO_FILE=$(ls -t "${VIDEO_DIR}/${BASE}"*.mp4 2>/dev/null | head -1)
if [ -z "$VIDEO_FILE" ]; then
  echo "[transcribe] error: no video file produced" >&2
  exit 1
fi
echo "[transcribe] video: $VIDEO_FILE"

# extract audio for whisper
echo "[transcribe] extracting audio"
AUDIO_FILE="${AUDIO_DIR}/${BASE}.wav"
ffmpeg -i "$VIDEO_FILE" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_FILE" 2>&1
echo "[transcribe] audio: $AUDIO_FILE"

# extract frames via scene detection
echo "[transcribe] extracting frames (scene threshold: $SCENE_THRESHOLD)"
ffmpeg -i "$VIDEO_FILE" \
  -vf "select='gt(scene,${SCENE_THRESHOLD})',scale=1280:-2" \
  -vsync vfr \
  "${FRAMES_DIR}/${BASE}_frame_%04d.png" 2>/dev/null

FRAME_COUNT=$(ls "${FRAMES_DIR}/${BASE}_frame_"*.png 2>/dev/null | wc -l)

# fallback: 1 frame every 10s if scene detection found nothing
if [ "$FRAME_COUNT" -eq 0 ]; then
  echo "[transcribe] scene detection yielded 0 frames — falling back to 1/10s interval"
  ffmpeg -i "$VIDEO_FILE" \
    -vf "fps=1/10,scale=1280:-2" \
    -vsync vfr \
    "${FRAMES_DIR}/${BASE}_frame_%04d.png" 2>/dev/null
  FRAME_COUNT=$(ls "${FRAMES_DIR}/${BASE}_frame_"*.png 2>/dev/null | wc -l)
fi

echo "[transcribe] frames extracted: $FRAME_COUNT"

# transcribe audio on GPU
echo "[transcribe] running whisper model=$MODEL"
whisper "$AUDIO_FILE" \
  --model "$MODEL" \
  --output_dir "$TRANSCRIPT_DIR" \
  --output_format all \
  --verbose False \
  2>&1

echo "[transcribe] done"
echo "[transcribe] transcript: ${TRANSCRIPT_DIR}/${BASE}.*"
echo "[transcribe] frames:     ${FRAMES_DIR}/${BASE}_frame_*.png ($FRAME_COUNT)"
