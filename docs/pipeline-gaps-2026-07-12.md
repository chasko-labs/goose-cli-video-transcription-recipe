# pipeline gaps + reinforcements (2026-07-12)

learnings from transcribing "Yells at Cloud Ep. 01" — 64-min, 5-speaker podcast, 1080p.

## gaps found

### 1. no speaker diarization

whisper transcribes words accurately but does not identify WHO is speaking. for a single-speaker tutorial this doesn't matter. for a 5-person podcast, it's the difference between a useful transcript and a wall of text.

**current state:** speaker attribution is entirely manual — done by reading context (names mentioned, topic expertise, conversational cues) and assigning speakers post-hoc.

**what's needed:** a diarization pass (pyannote.audio, NeMo, or whisperX) that segments audio by speaker voice and maps segments to speaker IDs. this should run between whisper and merge.

### 2. hs-transcribe timeout too short for long content

the default whisper timeout (900s) timed out at 35% on a 64-minute file. the pipeline reported "timeout" status while the download was still only partially complete (763MB at ~10MB/s = 76s download, then whisper medium on 64 min of audio).

**fix applied this session:** bypassed hs-transcribe entirely. hit the whisper-serve endpoint directly with pre-extracted audio:

```bash
curl -X POST http://localhost:8108/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "language=en" --max-time 1200 -o transcript.json
```

completed in 4m14s. the serve endpoint (large-v3-turbo) is dramatically faster than the docker container (medium model).

**what's needed:** hs-transcribe should scale timeout by file duration — e.g. `max(900, duration_seconds * 0.15)`. or better: detect that whisper-serve is available and route there instead of launching the docker container.

### 3. docker container entrypoint assumes full pipeline

the whisper docker image (`goose-cli-video-transcription-recipe-whisper:latest`) has an entrypoint that runs yt-dlp first. you cannot pass it a bare audio file — it tries to interpret the path as a URL.

**fix applied:** didn't use the docker image. used the serve endpoint instead.

**what's needed:** a `--audio-only-file` flag or a separate entrypoint that accepts pre-extracted audio. this would let you skip download+extract when those stages are already complete (resume pattern).

### 4. docker writes as root (permission trap)

the hs-transcribe pipeline creates `transcripts/` directory owned by root inside the output dir. subsequent non-root processes (like curl writing output) fail with permission denied.

**fix applied:** `sudo chown -R bryanchasko:heraldstack` before writing.

**what's needed:** the docker stages should run with `--user $(id -u):$(id -g)` or the orchestrator should fix permissions after each stage. this is documented in TRANSCRIBE.md but still bites every time.

### 5. no narrative quality gate

the pipeline produces a raw transcript and optionally a narrative via ollama. but there's no validation that the narrative is:

- complete (covers the full content, not truncated)
- accurate (names spelled correctly, services attributed properly)
- linked (companies and services have URLs)

**fix applied this session:** manual review caught 3 name misspellings (Gunnar Grosch, AJ Stuyvenberg, Danielle Heberling) that whisper phonetically mangled. web research corrected them.

**what's needed:** a post-narrative validation step that:

- checks named entities against a known-people database (AWS Heroes list, etc.)
- verifies completeness (word count of narrative vs. raw transcript ratio)
- flags phonetic name variants for human review

### 6. vision stage unnecessary for podcasts

373 frames extracted, each taking ~19s to analyze = ~2 hours of GPU time. for a talking-heads podcast, the vision stage adds almost no value — you get "person wearing blue shirt speaking into camera" repeated 373 times.

**what's needed:** content-type classification BEFORE vision dispatch. if the video is a podcast/video-call (detected by: uniform grid layout, face thumbnails, minimal scene changes), skip full vision and only analyze 3-5 manually-selected keyframes for thumbnails/screenshots.

### 7. whisper-serve vs docker: undocumented preference

the README and pipeline docs described the docker container path as primary. in practice, the serve endpoint is dramatically better: large-v3-turbo (15x realtime) vs. docker medium model (~realtime), zero startup time, no permission issues.

**fix applied:** created `docs/whisper-model.md` documenting the serve endpoint as preferred path.

**what's needed:** `transcribe-headless.sh` should check for whisper-serve availability FIRST and only fall back to docker if the endpoint is unreachable.

## reinforcements made

| what                                                                  | where                   | commit  |
| --------------------------------------------------------------------- | ----------------------- | ------- |
| whisper-large-v3-turbo model documentation                            | `docs/whisper-model.md` | 2c53049 |
| README architecture line updated (was "medium", now "large-v3-turbo") | `README.md`             | 2c53049 |
| first published blog transcript (proves the pipeline end-to-end)      | `blog/2026-07-12-*.md`  | 034febf |
| speaker attribution workflow documented                               | this file               | —       |
| serve endpoint as preferred path over docker                          | `docs/whisper-model.md` | 2c53049 |

## recommended pipeline changes (priority order)

1. **serve-first routing** — check whisper-serve:8108 health before launching docker. if healthy, POST audio directly. eliminates timeout issues, permission issues, and model-quality issues in one change.
2. **duration-scaled timeout** — if docker fallback is used, timeout = `max(900, audio_duration_seconds * 0.2)`.
3. **speaker diarization stage** — add between whisper and merge. pyannote.audio or whisperX. output: speaker-labeled segments.
4. **podcast detection** — classify content type from first 10 frames. if video-call/podcast, skip full vision, extract 3-5 key screenshots only.
5. **name verification** — post-narrative pass that checks named entities against searchable databases.
6. **user-match permissions** — docker stages run as calling user, not root.
