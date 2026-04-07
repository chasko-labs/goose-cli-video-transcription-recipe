# Video Transcription Recipe (goose-cli + ROCm)

Video transcription pipeline using goose-cli recipes on ROCm Docker. Whisper + Moondream2 + FFmpeg containerized on AMD GPU, triggered from Mac Mini via SSH.

Refactored from [kiro-cli-custom-agent-screenpal-video-transcription](https://github.com/chasko-labs/kiro-cli-custom-agent-screenpal-video-transcription).

## Architecture

Mac Mini (trigger) -> SSH -> ROCm-AIBOX (processing)

- Whisper: audio transcription (ROCm Docker container)
- Moondream2: visual frame analysis (Ollama on ROCm)
- FFmpeg: frame extraction with scene detection (Docker container)
- yt-dlp: media stream extraction
- goose-cli: recipe orchestration via goose-session

## Status

Refactor in progress on `refactor-to-goose` branch. See issues for backlog.

## Original Code

Original kiro-cli implementation preserved in `reference/kiro-original/` for context during refactor.
