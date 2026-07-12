FROM rocm/pytorch:latest

WORKDIR /app

# Add render (993) and video (44) groups to match host GIDs required for ROCm GPU access.
# Docker --group-add resolves names against the container's /etc/group; without these
# entries the container launch fails with "no matching entries in group file".
RUN groupadd -g 993 render 2>/dev/null || true && \
    groupadd -g 44 video 2>/dev/null || true

# whisper + yt-dlp + ffmpeg in one container
# yt-dlp extracts audio, whisper transcribes, ffmpeg handles format conversion
RUN pip install --no-cache-dir \
    openai-whisper \
    "yt-dlp==2026.03.17" \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pre-download medium model so container starts fast
RUN python3 -c "import whisper; whisper.load_model('medium')"

COPY transcribe.sh /app/transcribe.sh
RUN chmod +x /app/transcribe.sh

ENTRYPOINT ["/app/transcribe.sh"]
