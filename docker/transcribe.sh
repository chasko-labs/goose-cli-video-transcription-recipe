#!/bin/bash
# transcribe.sh — pipeline entrypoint
# usage: transcribe.sh <video-url> [model-size]
# output: timestamped transcript + frames in /media/
set -uo pipefail

URL="${1:?usage: transcribe.sh <video-url> [model-size]}"
MODEL="${2:-medium}"
SCENE_THRESHOLD="${SCENE_THRESHOLD:-0.4}"

AUDIO_DIR="/media/audio"
FRAMES_DIR="/media/frames"
TRANSCRIPT_DIR="/media/transcripts"
VIDEO_DIR="/media/videos"
mkdir -p "$AUDIO_DIR" "$FRAMES_DIR" "$TRANSCRIPT_DIR" "$VIDEO_DIR"

SLUG=$(echo "$URL" | sed 's|https\?://||;s|[^a-zA-Z0-9]|_|g' | cut -c1-60)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BASE="${TIMESTAMP}_${SLUG}"

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
  "${FRAMES_DIR}/${BASE}_frame_%04d.png" 2>&1
FRAME_COUNT=$(ls "${FRAMES_DIR}/${BASE}_frame_"*.png 2>/dev/null | wc -l)
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
