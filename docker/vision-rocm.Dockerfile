FROM rocm/pytorch:latest

WORKDIR /app

# transformers + vision deps — no model pre-download; HF cache is a volume mount
# latest transformers + tokenizers: tokenizer.json NFC normalizer requires tokenizers>=0.20
# which requires transformers>=4.45. apply_chunking_to_forward (removed in 4.45)
# is patched back in analyze-frames.py before the model loads.
RUN pip install --no-cache-dir \
    transformers \
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
