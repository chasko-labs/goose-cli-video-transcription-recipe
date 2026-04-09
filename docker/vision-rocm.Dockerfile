FROM rocm/pytorch:latest

WORKDIR /app

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
