#!/usr/bin/env python3
"""
analyze-frames.py — Instella-VL-1B frame analysis via ROCm
Usage: python3 analyze-frames.py <frames_dir> <base_prefix> <output_dir>
Output: <output_dir>/<base_prefix>-frame-analysis.json

Model: amd/Instella-VL-1B (VISION_MODEL env to override)
Architecture: CLIP ViT-L/14@336 + AMD OLMo 1B SFT + 2-layer MLP projector

Dependency conflict resolution:
  tokenizer.json NFC normalizer requires tokenizers>=0.20 which requires transformers>=4.45.
  But modeling_instellavl.py imports apply_chunking_to_forward removed in transformers 4.45.
  Patch: re-inject the function into transformers.modeling_utils before model load.
"""

import sys
import os
import json
import glob
import torch
import transformers.modeling_utils as _tmu
from pathlib import Path
from PIL import Image

# patch functions removed/moved from transformers.modeling_utils in 4.45+
# modeling_instellavl.py imports them from the old location

if not hasattr(_tmu, "apply_chunking_to_forward"):

    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            assert all(t.shape[chunk_dim] == tensor_shape for t in input_tensors)
            if tensor_shape % chunk_size != 0:
                raise ValueError(
                    f"Dimension to chunk ({tensor_shape}) must be a multiple of chunk_size ({chunk_size})"
                )
            num_chunks = tensor_shape // chunk_size
            chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
            output_chunks = tuple(forward_fn(*grp) for grp in zip(*chunks))
            return torch.cat(output_chunks, dim=chunk_dim)
        return forward_fn(*input_tensors)

    _tmu.apply_chunking_to_forward = _apply_chunking_to_forward

# find_pruneable_heads_and_indices + prune_linear_layer moved to pytorch_utils in 4.45+
try:
    from transformers.pytorch_utils import (
        find_pruneable_heads_and_indices,
        prune_linear_layer,
    )

    if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
        _tmu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(_tmu, "prune_linear_layer"):
        _tmu.prune_linear_layer = prune_linear_layer
except ImportError:
    pass

PROMPT = (
    "Describe all UI elements, text, menus, buttons, and visual content visible in this frame. "
    "Include exact text, labels, window titles, and interface details. Be specific and thorough."
)

MODEL_ID = os.environ.get("VISION_MODEL", "amd/Instella-VL-1B")


def load_model():
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[vision] loading {MODEL_ID}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    # encode() needs the CLIPImageProcessor from the vision tower (has crop_size)
    # not InstellaVLImageProcessor (processor.image_processor — no crop_size)
    clip_processor = model.get_model().get_vision_tower().image_processor
    print(f"[vision] {MODEL_ID} loaded on {device}", flush=True)
    return model, processor, clip_processor


def analyze_frame(model, processor, clip_processor, image_path):
    image = Image.open(image_path).convert("RGB")

    # InstellaVLProcessor.encode() handles conv template + image token insertion
    enc = processor.encode(
        text=PROMPT,
        images=image,
        image_processor=clip_processor,
        tokenizer=processor.tokenizer,
        model_cfg=model.config,
    )

    input_ids = enc["input_ids"].to("cuda")
    image_tensor = enc.get("image_tensor")
    image_sizes = enc.get("image_sizes")
    stopping_criteria = enc.get("stopping_criteria")
    eos_token_id = enc.get("eos_token_id")

    if image_tensor is not None:
        image_tensor = image_tensor.to("cuda", torch.float16)

    with torch.no_grad():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            max_new_tokens=300,
            do_sample=False,
            stopping_criteria=stopping_criteria,
            eos_token_id=eos_token_id,
        )

    # generate() uses inputs_embeds internally — output_ids is generated-only (no input prefix)
    generated = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.strip()


def main():
    if len(sys.argv) < 4:
        print("usage: analyze-frames.py <frames_dir> <base_prefix> <output_dir>")
        sys.exit(1)

    frames_dir, base_prefix, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    frames = sorted(glob.glob(f"{frames_dir}/{base_prefix}_frame_*.png"))
    if not frames:
        print(f"[vision] no frames found: {frames_dir}/{base_prefix}_frame_*.png")
        sys.exit(1)

    print(f"[vision] {len(frames)} frames to analyze", flush=True)

    model, processor, clip_processor = load_model()

    results = []
    for i, frame_path in enumerate(frames):
        frame_name = Path(frame_path).name
        frame_idx = int(frame_name.split("_frame_")[1].split(".")[0])
        print(f"[vision] {i + 1}/{len(frames)}: {frame_name}", flush=True)
        description = analyze_frame(model, processor, clip_processor, frame_path)
        results.append(
            {
                "frame": frame_name,
                "frame_index": frame_idx,
                "path": frame_path,
                "description": description,
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/{base_prefix}-frame-analysis.json"
    with open(out_path, "w") as f:
        json.dump(
            {"base": base_prefix, "model": MODEL_ID, "frames": results}, f, indent=2
        )

    print(f"[vision] done — {len(results)} frames → {out_path}")
    os._exit(0)  # skip pytorch/rocm cleanup crash during interpreter shutdown


if __name__ == "__main__":
    main()
