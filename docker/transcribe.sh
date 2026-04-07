#!/bin/bash
# transcribe.sh — pipeline entrypoint
# usage: transcribe.sh <video-url> [model-size]
# output: timestamped transcript in /media/transcripts/
set -uo pipefail

URL="${1:?usage: transcribe.sh <video-url> [model-size]}"
MODEL="${2:-medium}"

AUDIO_DIR="/media/audio"
FRAMES_DIR="/media/frames"
TRANSCRIPT_DIR="/media/transcripts"
mkdir -p "$AUDIO_DIR" "$FRAMES_DIR" "$TRANSCRIPT_DIR"

# derive a slug from the URL for filenames
SLUG=$(echo "$URL" | sed 's|https\?://||;s|[^a-zA-Z0-9]|_|g' | head -c 60)
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BASE="${TIMESTAMP}_${SLUG}"

echo "[transcribe] downloading audio from $URL"
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "${AUDIO_DIR}/${BASE}.%(ext)s" \
  "$URL" 2>&1 || {
    echo "[transcribe] yt-dlp failed — trying direct download"
    curl -sL "$URL" -o "${AUDIO_DIR}/${BASE}.wav"
  }

AUDIO_FILE=$(ls -t "${AUDIO_DIR}/${BASE}"*.wav 2>/dev/null | head -1)
if [ -z "$AUDIO_FILE" ]; then
  echo "[transcribe] error: no audio file produced" >&2
  exit 1
fi

echo "[transcribe] extracting frames (scene detection)"
ffmpeg -i "$AUDIO_FILE" -vn -acodec copy /dev/null 2>&1 || true
# if the source was a video, extract key frames
VIDEO_FILE=$(ls -t "${AUDIO_DIR}/${BASE}"*.mp4 "${AUDIO_DIR}/${BASE}"*.webm 2>/dev/null | head -1)
if [ -n "$VIDEO_FILE" ]; then
  ffmpeg -i "$VIDEO_FILE" \
    -vf "select='gt(scene,0.4)'" \
    -vsync vfr \
    "${FRAMES_DIR}/${BASE}_frame_%04d.png" 2>/dev/null || true
  echo "[transcribe] frames extracted to ${FRAMES_DIR}/"
fi

echo "[transcribe] running whisper model=$MODEL on $AUDIO_FILE"
whisper "$AUDIO_FILE" \
  --model "$MODEL" \
  --output_dir "$TRANSCRIPT_DIR" \
  --output_format all \
  --verbose False \
  2>&1

echo "[transcribe] done — outputs in ${TRANSCRIPT_DIR}/"
ls -la "${TRANSCRIPT_DIR}/${BASE}"* 2>/dev/null
