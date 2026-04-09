# video transcription pipeline (goose-cli + rocm)

5-stage video transcription pipeline with observability, resume, batch mode, and firecracker microvm dispatch. runs on amd gpu via rocm docker, triggered from mac mini over ssh.

refactored from [kiro-cli-custom-agent-screenpal-video-transcription](https://github.com/chasko-labs/kiro-cli-custom-agent-screenpal-video-transcription)

## architecture

```
mac mini (trigger) -> ssh -> rocm-aibox (processing)

stage 0: media-extract    fc-pool microvm (yt-dlp + ffmpeg) | fallback: whisper container
stage 1: whisper           gpu — audio transcription (medium model default)
stage 2: vision            gpu — instella-vl-1b frame analysis        } stages 1+2 run in parallel
stage 3: merge             correlate audio segments + visual frames by timestamp
stage 4: narrative         ollama llama3.1:8b weaves transcript + visuals into prose
```

## infrastructure

| component | location | purpose |
|---|---|---|
| rocm-aibox | 192.168.4.53 | gpu processing, docker, ollama, fc-pool |
| fc-pool | :8150 | firecracker microvm bridge for cpu-only media extraction |
| jaeger | :4318 (otlp), :16686 (ui) | distributed tracing — one trace per pipeline run |
| ollama | :11434 | narrative generation (llama3.1:8b) |

## usage

```bash
# single video
ssh rocm-aibox "bash ~/code/projects/goose-cli-video-transcription-recipe/scripts/transcribe-headless.sh <url> [model]"

# with flags
--force          re-run all stages (ignore existing outputs)
--dry-run        print plan without executing

# batch mode
--batch <file>   process urls from file (one per line, # comments ok)
--parallel <N>   concurrent pipelines in batch mode (default: 1)

# env overrides
TIMEOUT_STAGE2=3600  extend vision timeout for long videos
FC_POOL_URL=...      custom fc-pool endpoint
NARRATIVE_MODEL=...  ollama model override (default: llama3.1:8b)
```

## observability

- **status.json** — written at each stage boundary, pollable via `ssh rocm-aibox "cat ~/video-transcripts/<slug>/status.json"`
- **jaeger tracing** — otlp http json via scripts/trace.py, viewable at http://rocm-aibox.local:16686
- **per-stage timing** — wall-clock seconds printed after each stage completes
- **resume** — stages skip automatically when output files exist

## output structure

```
~/video-transcripts/
  <slug>/
    videos/          downloaded mp4
    audio/           extracted wav (16khz mono pcm)
    frames/          scene-detected or interval png frames
    transcripts/
      metadata.json                   yt-dlp metadata
      <base>.json                     whisper segments
      <base>-frame-analysis.json      vision descriptions
      <base>-combined.json            correlated audio + visual
      <base>-combined.md              human-readable merged doc
    status.json                       pipeline state
  narratives/
    <title-slug>.md                   llm-generated narrative
```

## scripts

| file | purpose |
|---|---|
| scripts/transcribe-headless.sh | main orchestrator — arg parsing, stage dispatch, timing, tracing, resume |
| scripts/batch.sh | batch mode orchestration (sourced by main script) |
| scripts/trace.py | otlp http json span sender for jaeger (stdlib only) |
| scripts/merge-outputs.py | stage 3 — correlate whisper segments + vision frames by timestamp |
| scripts/generate-narrative.py | stage 4 — build llm prompt, call ollama, write narrative md |

## docker images

| image | gpu | purpose |
|---|---|---|
| goose-cli-video-transcription-recipe-whisper | rocm | openai-whisper + yt-dlp + ffmpeg |
| goose-cli-video-transcription-recipe-vision | rocm | transformers + instella-vl-1b |

built automatically on first run if missing

## roadmap

- `--audio-only` flag for podcast transcription (skip vision, audio-only narrative prompt)
- tightener agent — post-processing stage that polishes llm narrative output (tighter prose, remove formulaic patterns, enforce style)
- fc-pool av1 codec support in microvm rootfs
- parallel batch mode e2e validation

## lineage

originally [kiro-cli-custom-agent-screenpal-video-transcription](https://github.com/chasko-labs/kiro-cli-custom-agent-screenpal-video-transcription). migrated to goose-cli recipes for headless ssh dispatch and recipe-based orchestration. original kiro-cli code preserved in `reference/kiro-original/`
