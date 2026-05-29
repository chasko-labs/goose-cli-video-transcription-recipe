# Pin to a known-stable rocm/pytorch tag.
# rocm/pytorch:latest is a floating tag — when AMD published rocm7.2.x + torch
# 2.10 (~2026-05) the auto-pulled latest broke Instella-VL-1B model load on
# RX 6700 XT (gfx1031) with `HIP error: out of memory` at .to("cuda") even
# with the full 12GB VRAM free. The regression is below the application
# layer (single-tensor allocation fails), not in our code or env vars.
# rocm6.4.4 + torch 2.6.0 is the last known-stable combination for this card.
FROM rocm/pytorch:rocm6.4.4_ubuntu22.04_py3.10_pytorch_release_2.6.0

WORKDIR /app

# Add render (993) and video (44) groups to match host GIDs required for ROCm GPU access.
# Docker --group-add resolves names against the container's /etc/group; without these
# entries the container launch fails with "no matching entries in group file".
RUN groupadd -g 993 render 2>/dev/null || true && \
    groupadd -g 44 video 2>/dev/null || true

# transformers + vision deps — no model pre-download; HF cache is a volume mount
# Pin to transformers==4.49.0 (AMD official supported version for Instella-VL-1B).
# - apply_chunking_to_forward removed in 4.45 → patched back in analyze-frames.py
# - find_pruneable_heads_and_indices moved to pytorch_utils in 4.44+ (present in 4.49)
# - tokenizers>=0.20 (NFC normalizer) compatible with transformers>=4.45 ✓
# Do NOT upgrade to 5.x: find_pruneable_heads_and_indices removed entirely there.
RUN pip install --no-cache-dir \
    "transformers==4.49.0" \
    accelerate \
    einops \
    pillow \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY analyze-frames.py /app/analyze-frames.py

# HuggingFace cache lives in a named volume — downloaded once, persisted across runs
ENV HF_HOME=/cache/huggingface
ENV VISION_MODEL=amd/Instella-VL-1B
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["python3", "/app/analyze-frames.py"]
