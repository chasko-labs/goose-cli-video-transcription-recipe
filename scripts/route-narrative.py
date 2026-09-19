#!/usr/bin/env python3
"""
route-narrative.py — vectorize a tightened narrative into qdrant gander-knowledge
(or an s3vectors index when ROUTE_BACKEND=s3vectors / --backend s3vectors).

usage:
  route-narrative.py <narrative-md-path> <video-dir> [--backend qdrant|s3vectors]
  route-narrative.py --help

backends:
  qdrant    (default) local qdrant gander-knowledge via fastembed MiniLM-384
  s3vectors amazon s3vectors index via bedrock titan-embed-1024.
            env: S3V_BUCKET (default typescript-course),
                 S3V_INDEX (default video-narratives),
                 S3V_REGION (default us-east-1),
                 S3V_EMBED_MODEL (default amazon.titan-embed-text-v2:0).
            boto3 resolves credentials from the standard chain
            (AWS_PROFILE honored).

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

observability:
  per-invocation structured log on stdout (jsonl), human warnings on stderr
  aggregator (last 100 invocations failure rate):
    tail -n 1000 output.log | grep '"event":"route_narrative"' | tail -n 100 | jq -s 'map(select(.qdrant_status=="fail")) | length / 100 * 100'
  live counters:
    docker exec heraldstack-valkey valkey-cli MGET gander:route:success gander:route:fail
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
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

_BACKEND = os.environ.get("ROUTE_BACKEND", "qdrant").strip().lower()
S3V_BUCKET = os.environ.get("S3V_BUCKET", "typescript-course")
S3V_INDEX = os.environ.get("S3V_INDEX", "video-narratives")
S3V_REGION = os.environ.get("S3V_REGION", "us-east-1")
S3V_EMBED_MODEL = os.environ.get("S3V_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
S3V_PUT_BATCH = 50

# chunking: target 2000-4000 chars (~500-1000 tokens @ 4 chars/token)
CHUNK_MIN = 2000
CHUNK_MAX = 4000

# deterministic namespace so uuid5 point ids are stable across runs
NAMESPACE = uuid.UUID("c7b2f6e8-6b0a-4e0c-9f1f-0e3a4d2b6a1a")


def log(msg: str) -> None:
    print(f"[route-narrative] {msg}", file=sys.stderr, flush=True)


def emit_structured_log(event: str, slug: str, chunk_count: int, status: str, ms: int, error: str | None = None, backend: str = "qdrant") -> None:
    """emit one-line jsonl structured log to stdout for downstream aggregation."""
    record = {
        "event": event,
        "slug": slug,
        "chunk_count": chunk_count,
        "backend": backend,
        "qdrant_status" if backend == "qdrant" else "s3vectors_status": status,
        "ms": ms,
    }
    if error:
        record["error"] = error
    print(json.dumps(record), flush=True)


def incr_valkey_counter(key: str) -> None:
    """increment a valkey counter via docker exec; non-fatal if valkey is down."""
    try:
        subprocess.run(
            ["docker", "exec", "heraldstack-valkey", "valkey-cli", "INCR", key],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        # valkey unavailable or docker exec failed — silent, non-fatal
        pass


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


S3V_CHUNK_MIN = 600
S3V_CHUNK_MAX = 1200  # keeps s3vectors filterable metadata under its 2048-byte cap


def chunk_narrative(body: str, chunk_min: int = CHUNK_MIN, chunk_max: int = CHUNK_MAX) -> list[str]:
    """split by h2/h3, then paragraph-split oversized chunks.

    narratives here rarely have headings — fallback is paragraph groups.
    s3vectors callers pass the smaller S3V_* bounds (metadata size cap).
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
        if len(s) > chunk_max:
            # flush buffer
            if buf:
                chunks.append(buf.strip())
                buf = ""
            # split oversized on paragraph then sentence
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
            sub = ""
            for para in paragraphs:
                if len(para) > chunk_max:
                    # sentence split
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sent in sentences:
                        if len(sub) + len(sent) + 1 > chunk_max and sub:
                            chunks.append(sub.strip())
                            sub = ""
                        sub += (" " if sub else "") + sent
                else:
                    if len(sub) + len(para) + 2 > chunk_max and sub:
                        chunks.append(sub.strip())
                        sub = ""
                    sub += ("\n\n" if sub else "") + para
            if sub:
                chunks.append(sub.strip())
            continue
        if len(buf) + len(s) + 2 <= chunk_max:
            buf += ("\n\n" if buf else "") + s
            if len(buf) >= chunk_min:
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


def video_id_from_narrative_path(path: Path) -> str:
    """video id = filename stem minus the '-narrative' suffix, case preserved."""
    stem = path.stem
    if stem.endswith("-narrative"):
        stem = stem[: -len("-narrative")]
    return stem


def s3v_clients():
    """boto3 clients for the s3vectors backend. import here so --help and
    the qdrant path work without boto3 installed."""
    import boto3
    from botocore.config import Config

    cfg = Config(
        retries={"total_max_attempts": 5, "mode": "adaptive"},
        connect_timeout=10,
        read_timeout=120,
    )
    bedrock = boto3.client("bedrock-runtime", region_name=S3V_REGION, config=cfg)
    s3v = boto3.client("s3vectors", region_name=S3V_REGION, config=cfg)
    return bedrock, s3v


def embed_texts_titan(bedrock, texts: list[str]) -> list[list[float]]:
    """embed via bedrock titan (one inputText per call)."""
    vecs: list[list[float]] = []
    for t in texts:
        resp = bedrock.invoke_model(
            modelId=S3V_EMBED_MODEL,
            body=json.dumps({"inputText": t}).encode("utf-8"),
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        vecs.append([float(x) for x in payload["embedding"]])
    return vecs


def s3v_existing_keys(s3v, video_id: str) -> list[str]:
    """keys previously written for this video.

    list_vectors has no server-side prefix/filter, so scan keys (cheap,
    no data/metadata) and match the video prefix client-side.
    """
    paginator = s3v.get_paginator("list_vectors")
    want = f"{video_id}-"
    keys: list[str] = []
    for page in paginator.paginate(
        vectorBucketName=S3V_BUCKET,
        indexName=S3V_INDEX,
        returnData=False,
        returnMetadata=False,
    ):
        keys.extend(v["key"] for v in page.get("vectors", []) if v["key"].startswith(want))
    return keys


def s3v_put_records(s3v, records: list[dict]) -> None:
    for i in range(0, len(records), S3V_PUT_BATCH):
        s3v.put_vectors(
            vectorBucketName=S3V_BUCKET,
            indexName=S3V_INDEX,
            vectors=records[i : i + S3V_PUT_BATCH],
        )


def route_narrative_s3v(narrative_path: Path, video_dir: Path) -> dict:
    start_ms = int(time.time() * 1000)
    md = narrative_path.read_text()
    meta, body = read_frontmatter(md)

    slug = slug_from_narrative_path(narrative_path)
    video_id = video_id_from_narrative_path(narrative_path)

    meta_json = read_metadata(find_metadata_json(video_dir))
    status_json = read_status(find_status_json(video_dir))

    source = detect_source(meta_json)
    try:
        duration_s = int(meta_json.get("duration") or 0)
    except (TypeError, ValueError):
        duration_s = 0
    whisper_model = (
        status_json.get("whisper_model")
        or os.environ.get("WHISPER_MODEL")
        or "unknown"
    )
    date_transcribed = (
        status_json.get("started_at")
        or status_json.get("date_transcribed")
        or datetime.now(timezone.utc).isoformat()
    )

    chunks = chunk_narrative(body, S3V_CHUNK_MIN, S3V_CHUNK_MAX)
    # hard cap per chunk: a single over-long sentence can exceed the chunker
    # bound; cap before embedding so vectors match stored text (2048B metadata cap)
    chunks = [c[:S3V_CHUNK_MAX] for c in chunks]
    if not chunks:
        log(f"skip: no chunks produced for {narrative_path.name}")
        ms = int(time.time() * 1000) - start_ms
        emit_structured_log("route_narrative", slug, 0, "ok", ms, backend="s3vectors")
        incr_valkey_counter("gander:route:success")
        return {"slug": slug, "chunks": 0, "skipped": True}

    log(f"s3vectors slug={slug} video={video_id} chunks={len(chunks)} source={source}")

    try:
        bedrock, s3v = s3v_clients()
        vectors = embed_texts_titan(bedrock, chunks)

        records = []
        for i, (text, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "key": f"{video_id}-{i:03d}",
                "data": {"float32": vec},
                "metadata": {
                    "slug": slug,
                    "video_id": video_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "title": meta.get("title", "")[:300],
                    "source": source,
                    "date_transcribed": str(date_transcribed),
                    "duration_s": duration_s,
                    "whisper_model": str(whisper_model),
                    "text": text,
                },
            })

        # idempotency: put fresh (same keys overwrite), then delete stale keys
        s3v_put_records(s3v, records)
        keep = {r["key"] for r in records}
        stale = [k for k in s3v_existing_keys(s3v, video_id) if k not in keep]
        if stale:
            s3v.delete_vectors(
                vectorBucketName=S3V_BUCKET,
                indexName=S3V_INDEX,
                keys=stale,
            )
            log(f"s3vectors pruned {len(stale)} stale keys for {video_id}")

        ms = int(time.time() * 1000) - start_ms
        emit_structured_log("route_narrative", slug, len(chunks), "ok", ms, backend="s3vectors")
        incr_valkey_counter("gander:route:success")
        return {"slug": slug, "chunks": len(chunks), "skipped": False}
    except Exception as e:
        ms = int(time.time() * 1000) - start_ms
        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
        emit_structured_log("route_narrative", slug, len(chunks), "fail", ms, error_msg, backend="s3vectors")
        incr_valkey_counter("gander:route:fail")
        raise


def route_narrative(narrative_path: Path, video_dir: Path) -> dict:
    start_ms = int(time.time() * 1000)
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
        ms = int(time.time() * 1000) - start_ms
        emit_structured_log("route_narrative", slug, 0, "ok", ms)
        incr_valkey_counter("gander:route:success")
        return {"slug": slug, "chunks": 0, "skipped": True}

    log(f"slug={slug} chunks={len(chunks)} source={source} dur={duration_s}s")

    try:
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
        ms = int(time.time() * 1000) - start_ms
        emit_structured_log("route_narrative", slug, len(chunks), "ok", ms)
        incr_valkey_counter("gander:route:success")
        return {"slug": slug, "chunks": len(chunks), "skipped": False}
    except Exception as e:
        ms = int(time.time() * 1000) - start_ms
        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
        emit_structured_log("route_narrative", slug, len(chunks), "fail", ms, error_msg)
        incr_valkey_counter("gander:route:fail")
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="route narrative to qdrant gander-knowledge (or s3vectors)")
    ap.add_argument("narrative", type=Path, help="path to tightened narrative .md")
    ap.add_argument("video_dir", type=Path, help="per-video directory with status.json + transcripts/metadata.json")
    ap.add_argument("--backend", choices=["qdrant", "s3vectors"], default=None,
                    help="vector backend (default: $ROUTE_BACKEND or qdrant)")
    args = ap.parse_args()

    backend = (args.backend or _BACKEND or "qdrant").strip().lower()
    if backend not in ("qdrant", "s3vectors"):
        log(f"error: unknown backend: {backend}")
        return 2

    if not args.narrative.is_file():
        log(f"error: narrative not found: {args.narrative}")
        return 2
    if not args.video_dir.is_dir():
        log(f"error: video_dir not found: {args.video_dir}")
        return 2

    try:
        if backend == "s3vectors":
            result = route_narrative_s3v(args.narrative, args.video_dir)
        else:
            result = route_narrative(args.narrative, args.video_dir)
    except Exception as e:
        log(f"error: {e}")
        # still return 0 to preserve non-fatal posture in pipeline hook
        return 0

    log(f"done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
