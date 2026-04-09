#!/usr/bin/env python3
"""
analyze-frames.py — Instella-VL-1B frame analysis via ROCm
Usage: python3 analyze-frames.py <frames_dir> <base_prefix> <output_dir>
Output: <output_dir>/<base_prefix>-frame-analysis.json

Model: amd/Instella-VL-1B (VISION_MODEL env to override)
Architecture: CLIP ViT-L/14@336 + AMD OLMo 1B SFT + 2-layer MLP projector
"""
import sys
import os
import json
import glob
import torch
from pathlib import Path
from PIL import Image

PROMPT = (
    "Describe all UI elements, text, menus, buttons, and visual content visible in this frame. "
    "Include exact text, labels, window titles, and interface details. Be specific and thorough."
)

MODEL_ID = os.environ.get("VISION_MODEL", "amd/Instella-VL-1B")


def load_model():
    # try modern auto class first (transformers 4.45+), fall back to LlavaForConditionalGeneration
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print(f"[vision] loading {MODEL_ID} via AutoModelForImageTextToText", flush=True)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
    except (ImportError, AttributeError, Exception) as e:
        print(f"[vision] auto class failed ({e}), falling back to LlavaForConditionalGeneration", flush=True)
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")

    model.eval()
    device = next(model.parameters()).device
    print(f"[vision] {MODEL_ID} loaded on {device}", flush=True)
    return model, processor


def analyze_frame(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")

    # LLaVA-style: try apply_chat_template; fall back to raw format string
    try:
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    except Exception:
        prompt_text = f"<image>\nUSER: {PROMPT}\nASSISTANT:"
        inputs = processor(text=prompt_text, images=image, return_tensors="pt")

    inputs = {k: v.to("cuda", torch.float16) if v.dtype.is_floating_point else v.to("cuda")
              for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300, do_sample=False)

    # decode only the generated portion
    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
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

    model, processor = load_model()

    results = []
    for i, frame_path in enumerate(frames):
        frame_name = Path(frame_path).name
        frame_idx = int(frame_name.split("_frame_")[1].split(".")[0])
        print(f"[vision] {i+1}/{len(frames)}: {frame_name}", flush=True)
        description = analyze_frame(model, processor, frame_path)
        results.append({
            "frame": frame_name,
            "frame_index": frame_idx,
            "path": frame_path,
            "description": description,
        })

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/{base_prefix}-frame-analysis.json"
    with open(out_path, "w") as f:
        json.dump({"base": base_prefix, "model": MODEL_ID, "frames": results}, f, indent=2)

    print(f"[vision] done — {len(results)} frames → {out_path}")


if __name__ == "__main__":
    main()
