#!/usr/bin/env python3
"""tidy.py — retention pass over ~/video-transcripts/

walks each <slug>/ subdir, applies the retention matrix from sprint ticket #21,
and either prints a dry-run table or removes eligible artifacts.

retention matrix:
  narratives/*.md                      -> forever (never touched)
  transcripts/*-combined.{json,md}     -> forever (never touched)
  transcripts/*.json (raw whisper)     -> 7d post-narrative (keep if no narrative)
  transcripts/*-frame-analysis.json    -> 7d post-narrative (keep if no narrative)
  transcripts/*.{srt,tsv,txt,vtt,tightened.json}  -> 7d post-narrative
  videos/*.mp4                         -> 7d post-narrative (always kept w/o narrative)
  audio/*.wav                          -> 7d post-narrative
  frames/*.png                         -> 7d post-narrative (keep if no narrative)

a narrative is considered to exist for a slug when any file under
<root>/narratives/*.md references one of that slug's *-combined base prefixes
(mirrors the combined_ref grep in transcribe-headless.sh line ~750).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# artifact subdirectories inside each <slug>/
SLUG_SUBDIRS = ("transcripts", "videos", "audio", "frames")

# directories to never touch, even recursively
PROTECTED_DIR_NAMES = {"narratives"}

# filename patterns (suffix match after base prefix) that are permanent
PROTECTED_TRANSCRIPT_SUFFIXES = ("-combined.json", "-combined.md")


@dataclass
class Candidate:
    path: Path
    slug: str
    category: str           # transcripts-raw, transcripts-frame, transcripts-aux, video, audio, frame
    age_days: float
    size_bytes: int
    keep_reason: Optional[str]  # None means eligible for removal


def _iter_slug_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in PROTECTED_DIR_NAMES:
            continue
        # only consider dirs that look like transcription outputs
        if any((child / sub).exists() for sub in SLUG_SUBDIRS):
            yield child


def _narrative_refs(narratives_dir: Path) -> set[str]:
    """return the set of combined_ref base prefixes mentioned across narratives/*.md.

    each narrative references its source combined file via a path containing
    '<base>-combined'. we return the set of <base> tokens.
    """
    refs: set[str] = set()
    if not narratives_dir.is_dir():
        return refs
    for md in narratives_dir.glob("*.md"):
        try:
            text = md.read_text(errors="replace")
        except OSError:
            continue
        # find any token ending in -combined — the immediately preceding
        # path-safe run of chars is the base prefix
        idx = 0
        while True:
            j = text.find("-combined", idx)
            if j == -1:
                break
            # walk backwards over path-safe chars to find base start
            k = j
            while k > 0 and (text[k - 1].isalnum() or text[k - 1] in "_-."):
                k -= 1
            base = text[k:j]
            if base:
                refs.add(base)
            idx = j + len("-combined")
    return refs


def _slug_has_narrative(slug_dir: Path, narrative_refs: set[str]) -> bool:
    """check whether any base prefix in this slug's transcripts/ appears in narrative_refs."""
    transcripts = slug_dir / "transcripts"
    if not transcripts.is_dir():
        return False
    for p in transcripts.iterdir():
        name = p.name
        if name.endswith("-combined.json") or name.endswith("-combined.md"):
            base = name[: -len("-combined.json")] if name.endswith("-combined.json") else name[: -len("-combined.md")]
            if base in narrative_refs:
                return True
    return False


def _age_days(p: Path, now: float) -> float:
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return 0.0
    return max(0.0, (now - mtime) / 86400.0)


def _classify(path: Path, slug_dir: Path) -> Optional[str]:
    """return the category token for a file, or None to skip it entirely."""
    try:
        rel = path.relative_to(slug_dir)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    bucket = parts[0]
    name = path.name

    if bucket == "transcripts":
        # protected forever
        for suf in PROTECTED_TRANSCRIPT_SUFFIXES:
            if name.endswith(suf):
                return None
        if name.endswith("-frame-analysis.json"):
            return "transcripts-frame"
        if name.endswith(".json"):
            return "transcripts-raw"
        # aux whisper outputs
        if name.endswith((".srt", ".tsv", ".txt", ".vtt")):
            return "transcripts-aux"
        return None
    if bucket == "videos" and name.endswith(".mp4"):
        return "video"
    if bucket == "audio" and name.endswith(".wav"):
        return "audio"
    if bucket == "frames" and name.endswith(".png"):
        return "frame"
    return None


def _walk_slug(
    slug_dir: Path,
    has_narrative: bool,
    now: float,
    older_than_days: int,
    keep_raw_transcripts: bool,
) -> list[Candidate]:
    out: list[Candidate] = []
    slug = slug_dir.name
    for sub in SLUG_SUBDIRS:
        d = slug_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            cat = _classify(p, slug_dir)
            if cat is None:
                continue
            age = _age_days(p, now)
            try:
                size = p.stat().st_size
            except OSError:
                size = 0

            keep_reason: Optional[str] = None
            if not has_narrative:
                keep_reason = "no narrative"
            elif age < older_than_days:
                keep_reason = f"age<{older_than_days}d"
            elif keep_raw_transcripts and cat in ("transcripts-raw", "transcripts-frame", "transcripts-aux"):
                keep_reason = "keep-raw"

            out.append(
                Candidate(
                    path=p,
                    slug=slug,
                    category=cat,
                    age_days=age,
                    size_bytes=size,
                    keep_reason=keep_reason,
                )
            )
    return out


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if x < 1024.0:
            return f"{x:6.1f}{unit}"
        x /= 1024.0
    return f"{x:6.1f}P"


def _print_table(rows: list[Candidate], home: Path) -> None:
    if not rows:
        print("(no candidate files found)")
        return
    # header
    print(f"{'path':<90} {'age_days':>9} {'reclaim':>8} keep_reason")
    print("-" * 90 + " " + "-" * 9 + " " + "-" * 8 + " " + "-" * 24)
    for r in rows:
        try:
            shown = "~/" + str(r.path.relative_to(home))
        except ValueError:
            shown = str(r.path)
        if len(shown) > 90:
            shown = "..." + shown[-87:]
        reclaim = 0 if r.keep_reason else r.size_bytes
        reason = r.keep_reason if r.keep_reason else "remove"
        print(f"{shown:<90} {r.age_days:>9.2f} {_fmt_bytes(reclaim):>8} {reason}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="tidy.py",
        description="retention pass over ~/video-transcripts/ (ticket #21)",
    )
    ap.add_argument(
        "--root",
        default=str(Path.home() / "video-transcripts"),
        help="root containing <slug>/ subdirs and narratives/ (default: ~/video-transcripts)",
    )
    ap.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    ap.add_argument("--older-than-days", type=int, default=7)
    ap.add_argument(
        "--keep-raw-transcripts",
        action="store_true",
        help="retain raw whisper + frame-analysis json even post-narrative",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: root not found: {root}", file=sys.stderr)
        return 2

    narratives_dir = root / "narratives"
    narrative_refs = _narrative_refs(narratives_dir)
    now = time.time()

    all_rows: list[Candidate] = []
    slugs_scanned = 0
    for slug_dir in _iter_slug_dirs(root):
        slugs_scanned += 1
        has_narr = _slug_has_narrative(slug_dir, narrative_refs)
        rows = _walk_slug(
            slug_dir,
            has_narr,
            now,
            args.older_than_days,
            args.keep_raw_transcripts,
        )
        all_rows.extend(rows)

    _print_table(all_rows, Path.home())

    files_kept = sum(1 for r in all_rows if r.keep_reason is not None)
    removable = [r for r in all_rows if r.keep_reason is None]
    bytes_reclaimable = sum(r.size_bytes for r in removable)

    files_removed = 0
    bytes_reclaimed = 0
    if args.mode == "apply":
        for r in removable:
            # belt-and-suspenders: never touch protected paths
            if any(part in PROTECTED_DIR_NAMES for part in r.path.parts):
                continue
            if r.path.name.endswith(PROTECTED_TRANSCRIPT_SUFFIXES):
                continue
            try:
                sz = r.size_bytes
                r.path.unlink()
                files_removed += 1
                bytes_reclaimed += sz
            except OSError as e:
                print(f"warn: could not remove {r.path}: {e}", file=sys.stderr)
    else:
        # dry-run: report what would be removed as the "reclaim" total
        bytes_reclaimed = bytes_reclaimable

    summary = {
        "mode": args.mode,
        "slugs_scanned": slugs_scanned,
        "files_kept": files_kept,
        "files_removed": files_removed if args.mode == "apply" else len(removable),
        "bytes_reclaimed": bytes_reclaimed,
    }
    print()
    print(
        "summary: "
        f"slugs_scanned={summary['slugs_scanned']} "
        f"files_kept={summary['files_kept']} "
        f"files_{'removed' if args.mode == 'apply' else 'removable'}={summary['files_removed']} "
        f"bytes_{'reclaimed' if args.mode == 'apply' else 'reclaimable'}={summary['bytes_reclaimed']} "
        f"({_fmt_bytes(summary['bytes_reclaimed']).strip()})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
