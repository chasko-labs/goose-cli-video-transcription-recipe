#!/usr/bin/env python3
"""intake.py — front-door dispatcher for batch transcription

accepts any of:
  - single video url
  - youtube/podcast playlist url (any yt-dlp playlist)
  - @channel handle (enumerate recent uploads)
  - file path containing one url per line (# comments + blank lines ok)

for each enumerated url:
  - derives the slug using the same rule as transcribe-headless.sh line ~257
  - dedups against ~/video-transcripts/<slug>/status.json
    (skip when stage_name == 'narrative' and status == 'done')
  - writes a manifest sidecar at ~/video-transcripts/.manifests/<date>-<label>.json
    [{url, slug, status, attempts, last_error}, ...]
  - writes a plain-text batch file alongside (one url per line) that is what
    actually gets handed to transcribe-headless.sh --batch

then unless --dry-run: dispatches scripts/transcribe-headless.sh --batch <txt>
--parallel N and, on completion, re-reads each per-slug status.json to build a
closing status table plus overwatch next-action hints.

complements #19 transcribe-local (single-url and pre-built batch file paths) —
this is the bulk / mixed-shape front door. no real transcription happens when
--dry-run is set; that flag exists so the recipe and helpers can be smoke-
tested without burning gpu time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DATA_DIR = Path.home() / "video-transcripts"
MANIFESTS_DIR = DATA_DIR / ".manifests"
REPO_DIR = Path(__file__).resolve().parent.parent
TRANSCRIBE_SCRIPT = REPO_DIR / "scripts" / "transcribe-headless.sh"

# matches the slug derivation in scripts/transcribe-headless.sh line ~257:
#   sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60
_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)
_WWW_RE = re.compile(r"^www\.", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")
_UNDER_RUN_RE = re.compile(r"_+")


def slug_for(url: str) -> str:
    s = _SCHEME_RE.sub("", url)
    s = _WWW_RE.sub("", s)
    s = _NON_ALNUM_RE.sub("_", s)
    s = _UNDER_RUN_RE.sub("_", s)
    s = s.strip("_")
    return s[:60]


# -- intake shape detection ---------------------------------------------------

# yt-dlp treats a watch?v= id as a single video even when it carries a list=
# playlist param. only treat the url as a playlist when the path itself is
# /playlist or the host is a podcast feed surfaced by yt-dlp.
_PLAYLIST_HINT_RE = re.compile(
    r"(?:youtube\.com/playlist\b|youtube\.com/.*[?&]list=|/podcast/|\.rss(?:\?|$))",
    re.IGNORECASE,
)
_VIDEO_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_HANDLE_RE = re.compile(r"^@[A-Za-z0-9_.-]+$")


def detect_shape(source: str) -> str:
    """return one of: file | handle | playlist | single"""
    p = Path(os.path.expanduser(source))
    if p.exists() and p.is_file():
        return "file"
    if _HANDLE_RE.match(source):
        return "handle"
    if _VIDEO_URL_RE.match(source):
        # playlist hints first — a /watch?v=X url with a list= is still a
        # single video for our purposes (yt-dlp would otherwise spider the
        # whole queue); the user can pass a /playlist?list= url explicitly
        if re.search(r"youtube\.com/playlist\b", source, re.IGNORECASE):
            return "playlist"
        if re.search(r"(?:/podcast/|\.rss(?:\?|$))", source, re.IGNORECASE):
            return "playlist"
        return "single"
    raise SystemExit(
        f"intake: could not classify source: {source!r} — expected url, @handle, or file path"
    )


def _ytdlp_flat(url: str) -> list[str]:
    """yt-dlp --flat-playlist -j <url> → list of resolved video urls.

    yt-dlp emits one json object per line with fields like id, url, ie_key.
    we reconstruct a canonical watch url from id when the record doesn't
    surface a full url (youtube channel/playlist output typically gives bare
    ids rather than urls).
    """
    cmd = ["yt-dlp", "--flat-playlist", "-j", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"intake: yt-dlp not on PATH — {exc}")
    if proc.returncode != 0:
        # surface stderr verbatim — caller (overwatch) maps common errors
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"intake: yt-dlp exited {proc.returncode} for {url}")
    out: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        u = rec.get("url") or rec.get("webpage_url")
        if not u and rec.get("id"):
            u = f"https://www.youtube.com/watch?v={rec['id']}"
        if u:
            out.append(u)
    return out


def _read_url_file(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        urls.append(line)
    return urls


def enumerate_urls(source: str, shape: str) -> list[str]:
    if shape == "single":
        return [source]
    if shape == "file":
        return _read_url_file(Path(os.path.expanduser(source)))
    if shape == "playlist":
        return _ytdlp_flat(source)
    if shape == "handle":
        handle = source.lstrip("@")
        return _ytdlp_flat(f"https://www.youtube.com/@{handle}/videos")
    raise AssertionError(f"unhandled shape: {shape}")


# -- dedup --------------------------------------------------------------------


def is_done(slug: str) -> bool:
    """skip when status.json shows stage_name=='narrative' and status=='done'."""
    status_path = DATA_DIR / slug / "status.json"
    if not status_path.exists():
        return False
    try:
        data = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("stage_name") == "narrative" and data.get("status") == "done"


# -- manifest -----------------------------------------------------------------


@dataclass
class ManifestRow:
    url: str
    slug: str
    status: str  # pending | skipped-dedup | done | retryable | failed
    attempts: int = 0
    last_error: str = ""


def build_manifest(urls: list[str]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        slug = slug_for(u)
        if is_done(slug):
            rows.append(ManifestRow(url=u, slug=slug, status="skipped-dedup"))
        else:
            rows.append(ManifestRow(url=u, slug=slug, status="pending"))
    return rows


def write_manifest(rows: list[ManifestRow], label: str) -> tuple[Path, Path]:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    date = _dt.date.today().isoformat()
    base = f"{date}-{label}"
    json_path = MANIFESTS_DIR / f"{base}.json"
    txt_path = MANIFESTS_DIR / f"{base}.txt"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    # txt: only non-skipped urls, one per line, for --batch dispatch
    txt_lines = [f"# generated by intake.py — {base}"]
    for r in rows:
        if r.status == "pending":
            txt_lines.append(r.url)
        else:
            txt_lines.append(f"# {r.status}: {r.url}")
    txt_path.write_text("\n".join(txt_lines) + "\n")
    return json_path, txt_path


# -- dispatch + overwatch -----------------------------------------------------


# common yt-dlp / pipeline failure signatures → next-action hint
OVERWATCH_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Sign in to confirm your age|age[- ]restricted", re.IGNORECASE),
     "yt-dlp age-gated -> paste cookies (yt-dlp --cookies-from-browser firefox)"),
    (re.compile(r"Private video|This video is private", re.IGNORECASE),
     "yt-dlp private video -> cannot auto-recover, remove url from batch"),
    (re.compile(r"Video unavailable", re.IGNORECASE),
     "video unavailable -> check url or regional block"),
    (re.compile(r"HTTP Error 429|rate[- ]limit", re.IGNORECASE),
     "yt-dlp rate-limited -> back off, retry in ~15min"),
    (re.compile(r"fc-pool.*(?:unreachable|refused|down)", re.IGNORECASE),
     "fc-pool down -> falling back to in-container whisper"),
    (re.compile(r"gpu_lock.*held|GPU_LOCK", re.IGNORECASE),
     "gpu semaphore contention -> reduce --parallel or wait"),
    (re.compile(r"CUDA out of memory|HIP out of memory", re.IGNORECASE),
     "gpu oom -> drop to smaller whisper model (small/base)"),
)


def overwatch_hints(blob: str) -> list[str]:
    hits: list[str] = []
    for pat, hint in OVERWATCH_RULES:
        if pat.search(blob) and hint not in hits:
            hits.append(hint)
    return hits


def dispatch_batch(txt_path: Path, parallel: int) -> tuple[int, str, str]:
    cmd = [
        "bash",
        str(TRANSCRIBE_SCRIPT),
        "--batch",
        str(txt_path),
        "--parallel",
        str(parallel),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # pass-through to our own stdio so the operator sees live-ish logs
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def classify_post_run(row: ManifestRow) -> ManifestRow:
    """after the batch, re-read each slug's status.json and classify."""
    status_path = DATA_DIR / row.slug / "status.json"
    if not status_path.exists():
        row.status = "failed"
        row.last_error = "no status.json written"
        return row
    try:
        data = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        row.status = "failed"
        row.last_error = f"status.json unreadable: {exc}"
        return row
    stage_name = data.get("stage_name", "")
    status = data.get("status", "")
    if stage_name == "narrative" and status == "done":
        row.status = "done"
    elif status == "failed":
        row.status = "retryable" if stage_name in {"media-extract", "vision"} else "failed"
        # look up the failed stage's detail, if any
        stages = data.get("stages", {})
        last = stages.get(str(data.get("stage", "")), {})
        row.last_error = last.get("error") or f"failed at {stage_name}"
    else:
        row.status = "retryable"
        row.last_error = f"stuck at {stage_name}:{status}"
    return row


_STATUS_GLYPH = {
    "done": "v",          # 'v' (checkmark-ish ascii) — per house style, no unicode
    "skipped-dedup": "=",
    "retryable": "o",
    "failed": "x",
    "pending": "-",
}


def print_closing_table(rows: list[ManifestRow]) -> None:
    print()
    print("closing status")
    print("-" * 78)
    print(f"{'':2}  {'status':14}  {'slug':40}  detail")
    print("-" * 78)
    for r in rows:
        glyph = _STATUS_GLYPH.get(r.status, "?")
        detail = r.last_error if r.last_error else r.url
        print(f"{glyph:2}  {r.status:14}  {r.slug[:40]:40}  {detail[:60]}")
    print("-" * 78)
    totals = {k: 0 for k in _STATUS_GLYPH}
    for r in rows:
        totals[r.status] = totals.get(r.status, 0) + 1
    print(
        f"total={len(rows)}  done={totals['done']}  retryable={totals['retryable']}  "
        f"failed={totals['failed']}  skipped={totals['skipped-dedup']}  "
        f"pending={totals['pending']}"
    )


# -- cli ----------------------------------------------------------------------


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="intake.py",
        description="front-door dispatcher for batch transcription (url, playlist, @handle, file)",
    )
    ap.add_argument("source", help="single url | playlist url | @handle | file path (one url per line)")
    ap.add_argument("--label", default=None,
                    help="manifest label suffix (default: ISO date only)")
    ap.add_argument("--parallel", type=int, default=2,
                    help="concurrent pipelines handed to transcribe-headless.sh (default 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate + write manifest, do not dispatch batch")
    return ap.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    shape = detect_shape(args.source)
    urls = enumerate_urls(args.source, shape)
    if not urls:
        print(f"intake: no urls enumerated from {args.source!r} (shape={shape})", file=sys.stderr)
        return 2
    rows = build_manifest(urls)

    label = args.label or shape
    json_path, txt_path = write_manifest(rows, label)
    pending = [r for r in rows if r.status == "pending"]
    skipped = [r for r in rows if r.status == "skipped-dedup"]

    print(f"intake: shape={shape} source={args.source}")
    print(f"intake: enumerated={len(urls)} unique={len(rows)} pending={len(pending)} skipped-dedup={len(skipped)}")
    print(f"intake: manifest: {json_path}")
    print(f"intake: batch:    {txt_path}")

    if args.dry_run:
        print("intake: --dry-run set, not dispatching")
        return 0

    if not pending:
        print("intake: nothing pending, skipping dispatch")
        print_closing_table(rows)
        return 0

    rc, stdout, stderr = dispatch_batch(txt_path, args.parallel)
    # post-run classification — mutate rows in place, skipping dedup rows
    for r in rows:
        if r.status == "pending":
            classify_post_run(r)
    # re-write manifest json with updated statuses
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")

    hints = overwatch_hints(stdout + "\n" + stderr)
    print_closing_table(rows)
    if hints:
        print()
        print("overwatch hints")
        print("-" * 78)
        for h in hints:
            print(f"- {h}")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
