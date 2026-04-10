# setup guide

step-by-step walkthrough for building this pipeline from scratch. each step is independently testable before wiring into the full pipeline

## prerequisites

- rocm-aibox or any linux host with amd gpu + rocm drivers
- docker with gpu passthrough (`--device=/dev/kfd --device=/dev/dri`)
- ollama running with llama3.1:8b pulled
- ssh access from mac mini (or any trigger machine)

## build order

### 1. whisper in docker — verify gpu transcription works

```bash
# build the whisper image
docker build -f docker/whisper-rocm.Dockerfile -t goose-cli-video-transcription-recipe-whisper:latest docker/

# test with a short video
mkdir -p /tmp/test-pipeline/{videos,audio,frames,transcripts}
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /tmp/test-pipeline/videos:/media/videos \
  -v /tmp/test-pipeline/audio:/media/audio \
  -v /tmp/test-pipeline/frames:/media/frames \
  -v /tmp/test-pipeline/transcripts:/media/transcripts \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  goose-cli-video-transcription-recipe-whisper:latest \
  "https://www.youtube.com/watch?v=YOUR_SHORT_VIDEO" medium
```

verify: `ls /tmp/test-pipeline/transcripts/*.json` shows whisper output with segments

### 2. frame extraction — verify scene detection

check `/tmp/test-pipeline/frames/` for png files from step 1. the container's transcribe.sh handles this automatically. if scene detection yields 0 frames, it falls back to 1 frame per 10 seconds

key variable: `SCENE_THRESHOLD=0.4` controls sensitivity. lower = more frames

### 3. vision analysis — single frame first

```bash
docker build -f docker/vision-rocm.Dockerfile -t goose-cli-video-transcription-recipe-vision:latest docker/

BASE=$(ls /tmp/test-pipeline/audio/*.wav | head -1 | xargs basename | sed 's/.wav//')
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /tmp/test-pipeline/frames:/media/frames \
  -v /tmp/test-pipeline/transcripts:/media/transcripts \
  -v goose-cli-video-transcription-recipe_hf-cache:/cache/huggingface \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e VISION_MODEL=amd/Instella-VL-1B \
  goose-cli-video-transcription-recipe-vision:latest \
  /media/frames "$BASE" /media/transcripts
```

verify: `*-frame-analysis.json` exists with frame descriptions

**gpu memory warning**: vision processes frames sequentially. 12gb vram handles ~50 frames before risk of oom. for videos with 100+ frames, consider TIMEOUT_STAGE2 override or frame subsampling

### 4. merge — correlate audio + visual

```bash
docker run --rm \
  -v /tmp/test-pipeline:/media \
  -v $(pwd)/scripts:/scripts \
  --entrypoint python3 \
  goose-cli-video-transcription-recipe-whisper:latest \
  /scripts/merge-outputs.py "$BASE" /media
```

verify: `*-combined.json` and `*-combined.md` exist. the md file should have frame-by-frame sections with visual descriptions matched to audio segments by timestamp

### 5. narrative — verify ollama integration

```bash
python3 scripts/generate-narrative.py /tmp/test-pipeline ~/video-transcripts/narratives
```

verify: markdown narrative in `~/video-transcripts/narratives/` with header metadata and 3-5 paragraph prose

requires: ollama running at localhost:11434 with llama3.1:8b

### 6. status.json + timing — makes everything debuggable

at this point, switch from manual stage execution to `transcribe-headless.sh`. the orchestrator adds:

- status.json written at every stage boundary
- wall-clock timing per stage
- `[pipeline] stage N done (Xs)` log lines

test: `cat ~/video-transcripts/<slug>/status.json` should show all stages with elapsed times

### 7. resume — idempotent re-runs

run the same url twice. second run should skip all stages:

```
[pipeline] stage 0: skipped (output exists)
[pipeline] stage 1: skipped (output exists)
...
```

use `--force` to override and re-run everything

### 8. audio-only — podcast support

```bash
ssh rocm-aibox "bash scripts/transcribe-headless.sh 'https://example.com/episode.mp3'"
```

auto-detection triggers from url pattern (.mp3, .wav). also infers from yt-dlp metadata (vcodec=none, podcast tags). skips vision, creates stub frame-analysis, runs audio-only narrative

### 9. tighten — filler removal

runs automatically after stages 1+2 (transcript) and after stage 4 (narrative). to test standalone:

```bash
# transcript pass (regex)
python3 scripts/tighten.py transcript /path/to/<base>.json

# narrative pass (ollama)
python3 scripts/tighten.py narrative /path/to/narrative.md
```

the regex pass strips: um, uh, you know, I mean, repeated words. merge prefers `<base>-tightened.json` over raw whisper output

### 10. tracing — jaeger integration

requires jaeger at :4318 (otlp http) and :16686 (ui). pipeline sends one trace per run with spans per stage

verify: open http://rocm-aibox.local:16686, search service `video-pipeline`

the trace sender is 94 lines of python using stdlib only (no opentelemetry sdk). deterministic trace IDs from slug + timestamp make traces reproducible

### 11. batch mode

```bash
cat > urls.txt << 'EOF'
https://www.youtube.com/watch?v=VIDEO1
https://www.youtube.com/watch?v=VIDEO2
# this line is a comment
EOF

ssh rocm-aibox "bash scripts/transcribe-headless.sh --batch urls.txt --parallel 2"
```

writes manifest to `~/video-transcripts/batch-<timestamp>.json` with per-url status and timing

### 12. fc-pool — optional, graduate-level

firecracker microvms for cpu-bound media extraction. reduces load on gpu host. not required — pipeline falls back to docker automatically

see [bryanchasko/heraldstack-firecracker](https://github.com/bryanchasko/heraldstack-firecracker) for setup

key considerations:
- requires kvm support, firecracker binary, jailer binary, kernel, rootfs images
- systemd service management (never start manually — orphan processes block port binding)
- snapshot/restore for <5s warm starts after ~40s cold boot
- av1 codec not supported in minimal rootfs (falls back to container)

## known constraints

- **12gb vram limit**: vision model + 100+ frames risks oom. use `TIMEOUT_STAGE2` env override for long videos, or accept audio-only fallback
- **av1 codec**: fc-pool microvm ffmpeg doesn't support av1. pipeline falls back to container (which does)
- **transcribe.sh expects video**: the container's entrypoint assumes .mp4 input. direct audio urls (.mp3) must use the audio-only download path, not the fallback container path
- **tighten narrative can expand**: the ollama tightening pass sometimes replaces terse phrases with longer "cleaner" versions on already-clean text. net effect is usually small
- **ollama required on host**: narrative generation and tightening call ollama at localhost:11434. not containerized — must be running on rocm-aibox

## design decisions

- **bash + python3 stdlib only**: no pip dependencies. python used for json manipulation, ollama http calls, otlp span construction. bash for orchestration, docker dispatch, timing
- **parallel subshells for gpu stages**: stages 1+2 run concurrently via `&` + `wait`. exit codes passed through temp files. fragile but functional — a separate-script-per-stage design would be cleaner
- **deterministic trace IDs**: sha256 of slug + timestamp. reproducible across re-runs, no uuid dependency
- **fire-and-forget tracing**: span sends are async (`&`), 5s timeout, never block the pipeline
- **atomic status.json**: python `os.rename` from tmp file. safe for concurrent readers polling via ssh
- **audio-only as inference, not just a flag**: url pattern, metadata vcodec, podcast tags, and frame count all feed the decision. explicit `--audio-only` overrides inference
