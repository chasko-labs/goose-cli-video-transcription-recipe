#!/usr/bin/env python3
"""
route-narrative.py — vectorize a tightened narrative into qdrant gander-knowledge

usage:
  route-narrative.py <narrative-md-path> <video-dir>
  route-narrative.py --help

reads:
  <narrative-md-path>            — the tightened narrative markdown
  <video-dir>/transcripts/metadata.json  — source / duration
  <video-dir>/status.json                — started_at (date_transcribed), whisper model

writes:
  qdrant collection gander-knowledge — one point per chunk (500-1000 tokens each)

idempotency:
  deletes all existing points where payload.slug == derived slug, then upserts fresh.

embedding:
  named vector `fast-all-minilm-l6-v2` (384-d), via fastembed + local qdrant REST.
  resolves python runtime from GANDER_ROUTE_VENV, else falls back to current.

exit codes:
  0 ok (or skipped with warning) — must be non-fatal to pipeline
  2 bad usage
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# re-exec under a venv if fastembed isn't available in the current interpreter.
# venv path is ~/.venvs/gander-route by default, override with GANDER_ROUTE_VENV.
def _ensure_venv_deps(entry_file: str | None = None) -> None:
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
        target = entry_file or __file__
        os.execve(str(venv_py), [str(venv_py), target, *sys.argv[1:]], env)


if __name__ == "__main__":
    _ensure_venv_deps()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("GANDER_KNOWLEDGE_COLLECTION", "gander-knowledge")
VECTOR_NAME = "fast-all-minilm-l6-v2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chunking: target 2000-4000 chars (~500-1000 tokens @ 4 chars/token)
CHUNK_MIN = 2000
CHUNK_MAX = 4000

# deterministic namespace so uuid5 point ids are stable across runs
NAMESPACE = uuid.UUID("c7b2f6e8-6b0a-4e0c-9f1f-0e3a4d2b6a1a")


def log(msg: str) -> None:
    print(f"[route-narrative] {msg}", file=sys.stderr, flush=True)


def read_frontmatter(md_text: str) -> tuple[dict, str]:
    """parse narrative frontmatter — the 'key: value' bullet lines before '---'"""
    lines = md_text.splitlines()
    meta: dict[str, str] = {}
    body_start = 0
    saw_header = False
    for i, line in enumerate(lines):
        if line.startswith("# ") and not saw_header:
            meta["title"] = line[2:].strip()
            saw_header = True
            continue
        if line.strip() == "---":
            body_start = i + 1
            break
        m = re.match(r"^\s*-\s*([a-z ]+):\s*(.*)$", line)
        if m:
            meta[m.group(1).strip().replace(" ", "_")] = m.group(2).strip()
    body = "\n".join(lines[body_start:]).strip()
    return meta, body


def slug_from_narrative_path(path: Path) -> str:
    """slug = filename stem, lowercased ascii"""
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem


def find_metadata_json(video_dir: Path) -> Path | None:
    p = video_dir / "transcripts" / "metadata.json"
    return p if p.is_file() else None


def find_status_json(video_dir: Path) -> Path | None:
    p = video_dir / "status.json"
    return p if p.is_file() else None


def detect_source(meta_json: dict | None) -> str:
    """enum: youtube.com/@goose-oss | podcast:<feed-url> | direct-audio | other"""
    if not meta_json:
        return "other"
    uploader_url = (meta_json.get("uploader_url") or "").lower()
    channel_url = (meta_json.get("channel_url") or "").lower()
    extractor = (meta_json.get("extractor") or "").lower()
    if "youtube.com/@goose-oss" in uploader_url or "youtube.com/@goose-oss" in channel_url:
        return "youtube.com/@goose-oss"
    # podcast heuristic — vcodec=none everywhere and rss-ish url
    webpage = (meta_json.get("webpage_url") or "").lower()
    if any(tok in webpage for tok in ("podcast", "/rss", "feed.xml")):
        return f"podcast:{meta_json.get('webpage_url','')}"
    if extractor == "youtube":
        return "other"
    if extractor in ("generic", "directaudio") and meta_json.get("acodec") and not meta_json.get("vcodec"):
        return "direct-audio"
    return "other"


def read_status(status_path: Path | None) -> dict:
    if not status_path:
        return {}
    try:
        return json.loads(status_path.read_text())
    except Exception as e:
        log(f"warning: failed to parse status.json: {e}")
        return {}


def read_metadata(meta_path: Path | None) -> dict:
    if not meta_path:
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        log(f"warning: failed to parse metadata.json: {e}")
        return {}


def chunk_narrative(body: str) -> list[str]:
    """split by h2/h3, then paragraph-split oversized chunks.

    narratives here rarely have headings — fallback is paragraph groups.
    """
    # first try heading split
    sections: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if re.match(r"^#{2,3}\s", line):
            if current:
                sections.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    # if single section, do paragraph grouping
    if len(sections) <= 1:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        sections = paragraphs

    # enforce size: merge tiny, split huge
    chunks: list[str] = []
    buf = ""
    for s in sections:
        if not s.strip():
            continue
        if len(s) > CHUNK_MAX:
            # flush buffer
            if buf:
                chunks.append(buf.strip())
                buf = ""
            # split oversized on paragraph then sentence
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
            sub = ""
            for para in paragraphs:
                if len(para) > CHUNK_MAX:
                    # sentence split
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        if len(sub) + len(sent) + 1 > CHUNK_MAX and sub:
                            chunks.append(sub.strip())
                            sub = ""
                        sub += (" " if sub else "") + sent
                else:
                    if len(sub) + len(para) + 2 > CHUNK_MAX and sub:
                        chunks.append(sub.strip())
                        sub = ""
                    sub += ("\n\n" if sub else "") + para
            if sub:
                chunks.append(sub.strip())
            continue
        if len(buf) + len(s) + 2 <= CHUNK_MAX:
            buf += ("\n\n" if buf else "") + s
            if len(buf) >= CHUNK_MIN:
                chunks.append(buf.strip())
                buf = ""
        else:
            if buf:
                chunks.append(buf.strip())
            buf = s
    if buf:
        chunks.append(buf.strip())

    # dedup empty
    return [c for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """embed via fastembed. imports inside fn so --help works without deps."""
    from fastembed import TextEmbedding  # type: ignore

    model = TextEmbedding(model_name=EMBED_MODEL)
    vecs = list(model.embed(texts))
    return [v.tolist() for v in vecs]


def qdrant_request(method: str, path: str, body: dict | None = None) -> dict:
    import urllib.request
    import urllib.error

    url = f"{QDRANT_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"qdrant {method} {path} -> {e.code}: {msg}") from e


def delete_by_slug(slug: str) -> int:
    """delete all points for this slug. returns count affected (best-effort)."""
    resp = qdrant_request(
        "POST",
        f"/collections/{COLLECTION}/points/delete?wait=true",
        {"filter": {"must": [{"key": "slug", "match": {"value": slug}}]}},
    )
    return 1 if resp.get("status") == "ok" else 0


def upsert_points(points: list[dict]) -> dict:
    return qdrant_request(
        "PUT",
        f"/collections/{COLLECTION}/points?wait=true",
        {"points": points},
    )


def route_narrative(narrative_path: Path, video_dir: Path) -> dict:
    md = narrative_path.read_text()
    meta, body = read_frontmatter(md)

    slug = slug_from_narrative_path(narrative_path)

    meta_json = read_metadata(find_metadata_json(video_dir))
    status_json = read_status(find_status_json(video_dir))

    source = detect_source(meta_json)
    duration_s = int(meta_json.get("duration") or 0)
    whisper_model = (
        status_json.get("whisper_model")
        or os.environ.get("WHISPER_MODEL")
        or "unknown"
    )
    date_transcribed = status_json.get("started_at") or datetime.now(timezone.utc).isoformat()

    chunks = chunk_narrative(body)
    if not chunks:
        log(f"skip: no chunks produced for {narrative_path.name}")
        return {"slug": slug, "chunks": 0, "skipped": True}

    log(f"slug={slug} chunks={len(chunks)} source={source} dur={duration_s}s")

    vectors = embed_texts(chunks)

    points = []
    for i, (text, vec) in enumerate(zip(chunks, vectors)):
        pid = str(uuid.uuid5(NAMESPACE, f"{slug}:{i}"))
        points.append({
            "id": pid,
            "vector": {VECTOR_NAME: vec},
            "payload": {
                "slug": slug,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "text": text,
                "title": meta.get("title", ""),
                "source": source,
                "date_transcribed": date_transcribed,
                "duration_s": duration_s,
                "whisper_model": whisper_model,
                "persona_interests": [],
            },
        })

    # idempotency: delete then upsert
    delete_by_slug(slug)
    upsert_points(points)
    return {"slug": slug, "chunks": len(chunks), "skipped": False}


def main() -> int:
    ap = argparse.ArgumentParser(description="route narrative to qdrant gander-knowledge")
    ap.add_argument("narrative", type=Path, help="path to tightened narrative .md")
    ap.add_argument("video_dir", type=Path, help="per-video directory with status.json + transcripts/metadata.json")
    args = ap.parse_args()

    if not args.narrative.is_file():
        log(f"error: narrative not found: {args.narrative}")
        return 2
    if not args.video_dir.is_dir():
        log(f"error: video_dir not found: {args.video_dir}")
        return 2

    try:
        result = route_narrative(args.narrative, args.video_dir)
    except Exception as e:
        log(f"error: {e}")
        return 1

    log(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
