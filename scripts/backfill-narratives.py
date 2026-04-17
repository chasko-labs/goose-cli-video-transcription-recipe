#!/usr/bin/env python3
"""
backfill-narratives.py — route every existing narrative into qdrant gander-knowledge

usage:
  backfill-narratives.py [narratives-dir] [transcripts-root]

defaults:
  narratives-dir   = ~/video-transcripts/narratives
  transcripts-root = ~/video-transcripts

for each narrative .md:
  1. parse the 'combined source:' frontmatter line to derive the video-dir
  2. if the video-dir exists, invoke route_narrative() directly
  3. otherwise skip with a warning

prints a summary {processed, skipped, failed}
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# venv re-exec (same semantics as route-narrative.py, but targets this script)
def _ensure_venv() -> None:
    if os.environ.get("_GANDER_ROUTE_REEXEC") == "1":
        return
    venv = os.environ.get("GANDER_ROUTE_VENV", str(Path.home() / ".venvs" / "gander-route"))
    venv_py = Path(venv) / "bin" / "python3"
    if not venv_py.is_file():
        return
    try:
        import fastembed  # noqa: F401
    except ModuleNotFoundError:
        env = dict(os.environ)
        env["_GANDER_ROUTE_REEXEC"] = "1"
        os.execve(str(venv_py), [str(venv_py), __file__, *sys.argv[1:]], env)


_ensure_venv()

# import the routing module
import importlib.util

route_spec = importlib.util.spec_from_file_location("route_narrative", HERE / "route-narrative.py")
assert route_spec and route_spec.loader
route_mod = importlib.util.module_from_spec(route_spec)
route_spec.loader.exec_module(route_mod)


COMBINED_SRC_RE = re.compile(r"^\s*-\s*combined source:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


def _video_id_from_combined(combined_path: Path) -> str | None:
    """extract the youtube id from a combined-source json filename, if possible."""
    # e.g. 20260409_035707_www_youtube_com_watch_v_8rTliYrQ6Iw-combined.json
    stem = combined_path.name
    m = re.search(r"watch_v_([A-Za-z0-9_-]+?)(?:-combined|\.|$)", stem)
    if m:
        return m.group(1)
    m = re.search(r"youtu_be_([A-Za-z0-9_-]+?)(?:-combined|\.|$)", stem)
    if m:
        return m.group(1)
    return None


def _find_video_dir_by_id(video_id: str, transcripts_root: Path) -> Path | None:
    """scan transcripts_root for any dir whose transcripts/metadata.json has matching id."""
    for cand in transcripts_root.iterdir():
        if not cand.is_dir():
            continue
        meta = cand / "transcripts" / "metadata.json"
        if not meta.is_file():
            continue
        try:
            d = json.loads(meta.read_text())
            if d.get("id") == video_id:
                return cand
        except Exception:
            continue
    return None


def video_dir_from_narrative(narr_path: Path, transcripts_root: Path) -> Path | None:
    """extract video-dir from 'combined source: <path>' frontmatter, or guess."""
    text = narr_path.read_text()
    m = COMBINED_SRC_RE.search(text)
    if m:
        combined = Path(m.group(1))
        # combined source is <video-dir>/transcripts/<stamp>-combined.json
        # so video-dir is two parents up
        if combined.parent.name == "transcripts":
            vd = combined.parent.parent
            if vd.is_dir():
                return vd
        # if the referenced dir was renamed, search by embedded video id
        vid_id = _video_id_from_combined(combined)
        if vid_id:
            found = _find_video_dir_by_id(vid_id, transcripts_root)
            if found:
                return found
    # fallback: guess from slug by looking at transcripts_root
    stem = narr_path.stem
    candidate = transcripts_root / stem
    if candidate.is_dir():
        return candidate
    return None


def main() -> int:
    narratives_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "video-transcripts" / "narratives"
    transcripts_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "video-transcripts"

    if not narratives_dir.is_dir():
        print(f"error: narratives-dir not found: {narratives_dir}", file=sys.stderr)
        return 2

    narratives = sorted(narratives_dir.glob("*.md"))
    if not narratives:
        print(f"no narratives under {narratives_dir}")
        return 0

    processed = 0
    skipped = 0
    failed = 0
    details: list[str] = []

    for narr in narratives:
        vd = video_dir_from_narrative(narr, transcripts_root)
        if vd is None:
            skipped += 1
            details.append(f"SKIP {narr.name} (no video-dir)")
            continue
        try:
            result = route_mod.route_narrative(narr, vd)
            processed += 1
            details.append(f"OK   {narr.name} chunks={result['chunks']}")
        except Exception as e:
            failed += 1
            details.append(f"FAIL {narr.name} {e}")

    for d in details:
        print(d)
    print()
    print(json.dumps({"processed": processed, "skipped": skipped, "failed": failed}))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
