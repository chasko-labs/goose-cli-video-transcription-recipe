#!/usr/bin/env python3
"""
merge-outputs.py — correlate whisper transcript + Instella-VL frame analysis
Usage: python3 merge-outputs.py <base_prefix> [media_root]

  base_prefix — timestamp-only, e.g. 20260409_035707
  media_root  — per-video directory containing audio/, frames/, transcripts/, videos/
                defaults to ~/video-transcripts/<inferred from cwd>

Reads:
  <media_root>/transcripts/<base_prefix>.json                   (whisper)
  <media_root>/transcripts/<base_prefix>-frame-analysis.json    (vision)
  <media_root>/videos/<base_prefix>.mp4                         (duration via ffprobe)
  <media_root>/transcripts/metadata.json                        (yt-dlp, optional)

Writes (persistent, re-runnable):
  <media_root>/transcripts/<base_prefix>-combined.json
  <media_root>/transcripts/<base_prefix>-combined.md

Frame timestamps are estimated evenly across video duration when no
sidecar timestamp file exists. ffprobe is used for duration if a video
file is present; falls back to last whisper segment end time.
"""
import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import timedelta


def fmt_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds - int(seconds)) * 10)
    return f"{h}:{m:02d}:{s:02d}.{ms}"


def get_video_duration(video_path: Path) -> float | None:
    """Return video duration in seconds via ffprobe, or None if unavailable."""
    if not video_path.exists():
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except Exception:
        pass
    return None


def assign_frame_timestamps(frames: list[dict], duration: float) -> list[dict]:
    """Estimate each frame's timestamp evenly across video duration."""
    n = len(frames)
    for i, frame in enumerate(frames):
        # +1 so frame 1 of 1 lands at 50% not 100%
        frame["timestamp_s"] = round((i + 0.5) / n * duration, 2)
    return frames


def find_segment_for_timestamp(segments: list[dict], ts: float) -> dict | None:
    """Return the whisper segment that contains ts, or nearest if none exact."""
    for seg in segments:
        if seg["start"] <= ts <= seg["end"]:
            return seg
    # nearest by midpoint distance
    if not segments:
        return None
    return min(segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - ts))


def build_combined(base: str, media_root: Path) -> dict:
    transcripts = media_root / "transcripts"
    whisper_path = transcripts / f"{base}.json"
    vision_path = transcripts / f"{base}-frame-analysis.json"
    video_path = media_root / "videos" / f"{base}.mp4"

    if not whisper_path.exists():
        raise FileNotFoundError(f"whisper JSON not found: {whisper_path}")
    if not vision_path.exists():
        raise FileNotFoundError(f"frame analysis JSON not found: {vision_path}")

    whisper = json.loads(whisper_path.read_text())
    vision = json.loads(vision_path.read_text())

    segments = whisper.get("segments", [])
    language = whisper.get("language", "unknown")
    frames = vision.get("frames", [])

    # get duration: ffprobe → last segment end → frame count estimate
    duration = get_video_duration(video_path)
    if duration is None and segments:
        duration = segments[-1]["end"]
    if duration is None:
        duration = len(frames) * 10.0  # fallback: assume 1 frame/10s
        print(f"[merge] no video file or segments — estimating duration={duration}s")
    else:
        print(f"[merge] video duration: {duration:.1f}s, segments: {len(segments)}, frames: {len(frames)}")

    frames = assign_frame_timestamps(frames, duration)

    combined_frames = []
    for frame in frames:
        ts = frame["timestamp_s"]
        seg = find_segment_for_timestamp(segments, ts)
        combined_frames.append({
            "frame": frame["frame"],
            "frame_index": frame["frame_index"],
            "timestamp_s": ts,
            "timestamp": fmt_ts(ts),
            "visual_description": frame.get("description", ""),
            "audio_segment": {
                "start": seg["start"] if seg else None,
                "end": seg["end"] if seg else None,
                "text": seg["text"].strip() if seg else "",
            } if seg else None,
        })

    full_transcript = whisper.get("text", "").strip()

    return {
        "base": base,
        "language": language,
        "duration_s": duration,
        "vision_model": vision.get("model", ""),
        "total_frames": len(frames),
        "total_segments": len(segments),
        "full_transcript": full_transcript,
        "frames": combined_frames,
    }


def write_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[merge] wrote {path}")


def write_md(data: dict, path: Path):
    lines = []
    lines.append(f"# {data['base']}")
    lines.append("")
    lines.append(f"- duration: {data['duration_s']:.1f}s")
    lines.append(f"- language: {data['language']}")
    lines.append(f"- frames analyzed: {data['total_frames']}")
    lines.append(f"- whisper segments: {data['total_segments']}")
    lines.append(f"- vision model: {data['vision_model']}")
    lines.append("")

    if data["full_transcript"]:
        lines.append("## full transcript")
        lines.append("")
        lines.append(data["full_transcript"])
        lines.append("")

    lines.append("## frame-by-frame")
    lines.append("")

    for f in data["frames"]:
        lines.append(f"### frame {f['frame_index']} — {f['timestamp']}")
        lines.append("")
        visual = f.get("visual_description", "")
        if visual:
            lines.append(f"**visual:** {visual}")
        else:
            lines.append("**visual:** *(no description)*")
        lines.append("")
        seg = f.get("audio_segment")
        if seg and seg.get("text"):
            lines.append(f"**audio [{fmt_ts(seg['start'])}–{fmt_ts(seg['end'])}]:** {seg['text']}")
        else:
            lines.append("**audio:** *(no speech)*")
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"[merge] wrote {path}")


def main():
    if len(sys.argv) < 2:
        print("usage: merge-outputs.py <base_prefix> [media_root]")
        sys.exit(1)

    base = sys.argv[1]
    # media_root is the per-video dir (contains transcripts/, videos/, etc.)
    media_root = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path("/media")

    data = build_combined(base, media_root)

    transcripts = media_root / "transcripts"
    write_json(data, transcripts / f"{base}-combined.json")
    write_md(data, transcripts / f"{base}-combined.md")

    print(f"[merge] done — {data['total_frames']} frames, {data['total_segments']} segments")


if __name__ == "__main__":
    main()
