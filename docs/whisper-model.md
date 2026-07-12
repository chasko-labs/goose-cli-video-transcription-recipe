# whisper model — large-v3-turbo

the pipeline uses **openai/whisper-large-v3-turbo** via a persistent serve endpoint (`whisper-serve.py` on port 8108, hs-shannon systemd service).

## what is large-v3-turbo?

OpenAI's distilled variant of whisper-large-v3 — same accuracy on English transcription but ~3x faster inference. it achieves this with 4 decoder layers instead of 32, cutting decode time dramatically while preserving the full encoder (which does most of the acoustic work). released late 2024, it's the best speed/quality tradeoff for English content.

## model comparison

| model                                | params | VRAM (fp16) | speed (64-min video, RX 6700 XT) | quality                |
| ------------------------------------ | ------ | ----------- | -------------------------------- | ---------------------- |
| **whisper-large-v3-turbo** (primary) | ~809M  | ~2.5GB      | **4m 14s** (~15x realtime)       | near-large-v3          |
| whisper-large-v3                     | ~1.5B  | ~4GB        | ~10-12min                        | best                   |
| whisper-medium (fallback)            | ~769M  | ~2.5GB      | ~6-8min                          | good                   |
| whisper-small                        | ~244M  | ~1GB        | ~2-3min                          | acceptable             |
| whisper-base                         | ~74M   | ~0.5GB      | <1min                            | poor for multi-speaker |

large-v3-turbo gives us large-v3 quality at medium-model speed. there is no reason to use medium anymore except as a fallback if turbo fails to load.

## serve endpoint

the model runs as a persistent systemd service (`hs-shannon` user), loaded once at startup, stays resident in VRAM.

```
endpoint:   POST http://localhost:8108/v1/audio/transcriptions
format:     multipart/form-data (OpenAI-compatible)
fields:     file (audio bytes), language (optional, e.g. "en")
returns:    {"text": "full transcription..."}
VRAM:       ~2.5GB resident (fp16)
concurrency: serialized (asyncio.Lock) — one request at a time
```

### usage

```bash
curl -X POST http://localhost:8108/v1/audio/transcriptions \
  -F "file=@path/to/audio.wav" \
  -F "language=en" \
  -o transcript.json
```

### from the pipeline

the `hs-transcribe` orchestrator calls this endpoint automatically. the docker container fallback is only used if the serve endpoint is unreachable.

## performance (measured 2026-07-12)

test: "Yells at Cloud Ep. 01" — 64m 32s, 1080p, 5 speakers, conversational podcast.

| stage                            | method                   | time       | notes                                  |
| -------------------------------- | ------------------------ | ---------- | -------------------------------------- |
| download (yt-dlp)                | network                  | ~90s       | 763MB 1080p                            |
| audio extraction (ffmpeg)        | cpu                      | ~12s       | 312x realtime, 16kHz mono WAV          |
| frame extraction (scene detect)  | cpu                      | ~30s       | 373 scene-detected keyframes           |
| **whisper transcription**        | **gpu (large-v3-turbo)** | **4m 14s** | **15x realtime, 65K chars, 12K words** |
| vision analysis (instella-vl-1b) | gpu                      | ~19s/frame | ~117 min for 373 frames                |
| merge                            | cpu                      | <5s        | timestamp alignment                    |
| narrative (ollama)               | gpu                      | ~30-60s    | llama3.1:8b synthesis                  |

### total wall time for 64-min video

- **whisper-only** (download → extract → transcribe): **~6 min**
- **full pipeline** (all stages including vision): **~2.5 hours** (vision dominates)

## fallback behavior

if `whisper-serve.py` is unreachable (port 8108 not responding):

1. pipeline logs warning
2. falls back to docker container (`goose-cli-video-transcription-recipe-whisper:latest`)
3. container loads whisper-medium (not turbo — different image)
4. slower startup (~30-60s model load) + slower inference (~realtime)
5. VRAM released after container exits

## VRAM budget

the RX 6700 XT has 12GB. whisper-large-v3-turbo uses ~2.5GB resident. this leaves room for:

- instella-vl-1b serve (vision): ~2.5GB — coexists fine
- ollama (llama3.1:8b): ~5GB — tight but works if ComfyUI is stopped
- ComfyUI (SDXL): ~4-6GB — CONFLICTS with ollama. stop one to run the other.

if running whisper + vision + ollama simultaneously: ~10GB of 12GB = safe.
if ComfyUI is also loaded: OOM. stop comfyui (`sudo systemctl stop comfyui`) before running the full pipeline.

check with: `rocm-smi --showpids` + `rocm-smi --showmeminfo vram`
