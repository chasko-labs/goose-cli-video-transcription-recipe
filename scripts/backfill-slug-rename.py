#!/usr/bin/env python3
"""
backfill-slug-rename.py — rename date-stamped narrative files to content-derived slugs
                          and re-route qdrant points accordingly.

usage:
  backfill-slug-rename.py [narratives-dir] [transcripts-root]

- reads RENAME_MAP below for explicit old→new slug assignments
- renames narrative .md file (does not modify content)
- if old slug exists in qdrant: deletes those points, upserts under new slug
- if new slug already exists as a file: logs conflict and skips
- duplicate slugs (where the target file already exists with correct content): just deletes old qdrant points
- prints summary {renamed, rerouted, skipped, deleted_duplicates}
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


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

import importlib.util

route_spec = importlib.util.spec_from_file_location("route_narrative", HERE / "route-narrative.py")
assert route_spec and route_spec.loader
route_mod = importlib.util.module_from_spec(route_spec)
route_spec.loader.exec_module(route_mod)

backfill_spec = importlib.util.spec_from_file_location("backfill_narratives", HERE / "backfill-narratives.py")
assert backfill_spec and backfill_spec.loader
backfill_mod = importlib.util.module_from_spec(backfill_spec)
backfill_spec.loader.exec_module(backfill_mod)


# ---------------------------------------------------------------------------
# explicit rename map: old_slug → (new_slug, action)
# action: "rename"   — rename file + re-route qdrant
#         "duplicate" — target file already exists; delete old qdrant points only
#         "keep"      — date-stamp is appropriate; skip
# ---------------------------------------------------------------------------
RENAME_MAP: dict[str, tuple[str, str]] = {
    "20260409-143812-www-youtube-com-watch-v-dqw4w9wgxcq":
        ("rick-astley-never-gonna-give-you-up-4k-remaster", "rename"),
    "20260409-214444-www-youtube-com-watch-v-1arftywlbao":
        ("goose-v1260-local-inference-telegram-gateway-peekaboo-vision", "duplicate"),
    "20260411-204837-www-youtube-com-watch-v-yexm6uvwg-q":
        ("chrome-extension-custom-side-panels", "rename"),
    "20260414-171000-www-youtube-com-watch-v-htd98mepu4s":
        ("woodpecker-ci-docker-windows-grandad-does-stuff", "rename"),
    "20260414-192818-youtu-be-38juakz6m5s-si-30z0pmonqysnckkg":
        ("20260414-192818-youtu-be-38juakz6m5s-si-30z0pmonqysnckkg", "keep"),
    "20260415-180802-www-youtube-com-watch-v-u-jl7bzab8a":
        ("sub-agents-codex-essentials-andrew", "rename"),
    "20260417-175814-www-youtube-com-watch-v-npuxzx5rdjk":
        ("goose-ai-tool-demo-sebastian", "rename"),
    "20260417-203621-www-youtube-com-watch-v-dg1hufsekyc":
        ("goose-neighborhood-extension-meal-ordering", "rename"),
}

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("GANDER_KNOWLEDGE_COLLECTION", "gander-knowledge")


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


def slug_in_qdrant(slug: str) -> bool:
    resp = qdrant_request(
        "POST",
        f"/collections/{COLLECTION}/points/count",
        {"filter": {"must": [{"key": "slug", "match": {"value": slug}}]}},
    )
    return resp.get("result", {}).get("count", 0) > 0


def delete_by_slug(slug: str) -> None:
    qdrant_request(
        "POST",
        f"/collections/{COLLECTION}/points/delete?wait=true",
        {"filter": {"must": [{"key": "slug", "match": {"value": slug}}]}},
    )


def main() -> int:
    narratives_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "video-transcripts" / "narratives"
    transcripts_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "video-transcripts"

    renamed = 0
    rerouted = 0
    skipped = 0
    deleted_duplicates = 0
    details: list[str] = []

    for old_slug, (new_slug, action) in RENAME_MAP.items():
        old_path = narratives_dir / f"{old_slug}.md"
        new_path = narratives_dir / f"{new_slug}.md"

        if action == "keep":
            skipped += 1
            details.append(f"KEEP  {old_slug}")
            continue

        if not old_path.exists():
            skipped += 1
            details.append(f"SKIP  {old_slug} (file not found)")
            continue

        if action == "duplicate":
            # target file should already exist; delete old qdrant points
            if not new_path.exists():
                details.append(f"WARN  {old_slug} duplicate target missing {new_path.name} — skipping")
                skipped += 1
                continue
            in_qdrant = slug_in_qdrant(old_slug)
            if in_qdrant:
                delete_by_slug(old_slug)
                details.append(f"DEL   {old_slug} (duplicate of {new_slug}, qdrant points deleted)")
            else:
                details.append(f"DEL   {old_slug} (duplicate of {new_slug}, not in qdrant)")
            old_path.unlink()
            deleted_duplicates += 1
            continue

        # action == "rename"
        if new_path.exists():
            details.append(f"CONF  {old_slug} → {new_slug} (target exists, skipping rename)")
            skipped += 1
            continue

        # do the rename
        old_path.rename(new_path)
        renamed += 1
        details.append(f"MV    {old_slug} → {new_slug}")

        # update narrative H1 in new file so title matches new slug
        text = new_path.read_text()
        # replace the first H1 line (the URL-derived one) with the new slug as title
        # convert slug → display title: hyphens → spaces, title-case
        display = new_slug.replace("-", " ").title()
        text = re.sub(r"^# .*$", f"# {display}", text, count=1, flags=re.MULTILINE)
        new_path.write_text(text)

        # re-route qdrant: delete old slug points, upsert new
        in_qdrant = slug_in_qdrant(old_slug)
        if in_qdrant:
            delete_by_slug(old_slug)
            # find video dir for this narrative
            vd = backfill_mod.video_dir_from_narrative(new_path, transcripts_root)
            if vd is None:
                details.append(f"WARN  {new_slug} no video-dir for qdrant re-route")
            else:
                try:
                    result = route_mod.route_narrative(new_path, vd)
                    rerouted += 1
                    details.append(f"ROUTE {new_slug} chunks={result['chunks']}")
                except Exception as e:
                    details.append(f"FAIL  {new_slug} qdrant route error: {e}")
        else:
            details.append(f"NOTE  {new_slug} not in qdrant (file renamed only)")

    for d in details:
        print(d)
    print()
    summary = {"renamed": renamed, "rerouted": rerouted, "skipped": skipped, "deleted_duplicates": deleted_duplicates}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
