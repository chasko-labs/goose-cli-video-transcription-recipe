#!/usr/bin/env python3
"""
generate-narrative.py — weave whisper transcript + frame analysis into a narrative
Usage: python3 generate-narrative.py <video_dir> [narratives_dir] [model]

  video_dir      — per-video dir, e.g. ~/video-transcripts/youtube_com_watch_v_8rTliYrQ6Iw
                   must contain transcripts/<timestamp>-combined.json and transcripts/metadata.json
  narratives_dir — output dir (default: ~/video-transcripts/narratives)
  model          — ollama model (default: llama3.1:8b or $NARRATIVE_MODEL env)

Reads:
  <video_dir>/transcripts/<latest>-combined.json
  <video_dir>/transcripts/metadata.json   (optional, enriches prompt)

Writes:
  <narratives_dir>/<content-title-slug>.md
  (filename derived from video title in metadata, not from URL)

Calls Ollama at http://localhost:11434 — must be running on the host.
"""
import sys
import os
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


NARRATIVE_MODEL = os.environ.get("NARRATIVE_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


def find_latest_combined(transcripts_dir: Path) -> Path | None:
    candidates = sorted(transcripts_dir.glob("*-combined.json"), reverse=True)
    return candidates[0] if candidates else None


def load_metadata(transcripts_dir: Path) -> dict:
    meta_path = transcripts_dir / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return {}


def build_prompt(combined: dict, meta: dict) -> tuple[str, str]:
    """Return (prompt, content_title) for the narrative."""
    title = meta.get("title") or combined.get("base", "untitled")
    uploader = meta.get("uploader") or meta.get("channel") or "unknown"
    description = (meta.get("description") or "")[:500]
    tags = ", ".join(meta.get("tags", [])[:10])

    full_transcript = combined.get("full_transcript", "").strip()
    frames = combined.get("frames", [])

    # summarise frames: timestamp + first 120 chars of visual description
    frame_lines = []
    for f in frames:
        desc = (f.get("visual_description") or "").strip()
        if desc:
            frame_lines.append(f"  [{f['timestamp']}] {desc[:120]}")
    frames_summary = "\n".join(frame_lines) if frame_lines else "(no frame descriptions)"

    prompt = f"""You are writing a narrative summary of a video for a knowledge base.

Video metadata:
- Title: {title}
- Creator/Channel: {uploader}
- Description: {description}
- Tags: {tags}
- Duration: {combined.get('duration_s', 0):.0f}s

Full audio transcript:
{full_transcript or "(no speech detected)"}

Visual frame analysis (timestamp + what is visible on screen):
{frames_summary}

Write a narrative that tells the story of this video in 3–5 paragraphs. Weave the audio content with the visual descriptions — describe what the speaker is showing or demonstrating at key moments. Write in third person. Refer to the creator/presenter by name if evident. Be specific about what is covered, shown, and demonstrated. Do not bullet-list — write flowing prose. End with a one-sentence summary of who would benefit from watching."""

    return prompt, title


def call_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 800},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("response", "").strip()


def main():
    if len(sys.argv) < 2:
        print("usage: generate-narrative.py <video_dir> [narratives_dir] [model]")
        sys.exit(1)

    video_dir = Path(sys.argv[1]).expanduser()
    narratives_dir = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 \
        else Path.home() / "video-transcripts" / "narratives"
    model = sys.argv[3] if len(sys.argv) > 3 else NARRATIVE_MODEL

    transcripts_dir = video_dir / "transcripts"
    combined_path = find_latest_combined(transcripts_dir)
    if not combined_path:
        print(f"[narrative] no combined.json found in {transcripts_dir}")
        sys.exit(1)

    combined = json.loads(combined_path.read_text())
    meta = load_metadata(transcripts_dir)

    print(f"[narrative] building prompt from {combined_path.name}", flush=True)
    prompt, title = build_prompt(combined, meta)

    print(f"[narrative] calling {model} via ollama", flush=True)
    try:
        narrative = call_ollama(prompt, model)
    except urllib.error.URLError as e:
        print(f"[narrative] ollama unreachable at {OLLAMA_URL}: {e}")
        sys.exit(1)

    narratives_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
    out_path = narratives_dir / f"{slug}.md"

    # if slug already exists from a prior run, append timestamp to avoid overwrite
    if out_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = narratives_dir / f"{slug}-{ts}.md"

    header = f"# {title}\n\n"
    header += f"- source: {meta.get('webpage_url', meta.get('original_url', ''))}\n"
    header += f"- creator: {meta.get('uploader', meta.get('channel', 'unknown'))}\n"
    header += f"- upload date: {meta.get('upload_date', 'unknown')}\n"
    header += f"- model: {model}\n"
    header += f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    header += f"- combined source: {combined_path}\n\n---\n\n"

    out_path.write_text(header + narrative + "\n")
    print(f"[narrative] wrote {out_path}")


if __name__ == "__main__":
    main()
