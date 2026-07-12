---
name: t-02 gpu semaphore verification
description: gate results for valkey gpu_lock semaphore in scripts/transcribe-headless.sh, branch refactor-to-goose, 2026-04-14
type: project
---

All five gates passed on 2026-04-14 on branch refactor-to-goose.

key: gpu_lock, TTL: 1200, transport: docker exec heraldstack-valkey valkey-cli

- gate A (happy-path acquire+release): pass. acquire rc=0, GET matches GPU_LOCK_OWNER, TTL=1200, EXISTS=0 after release
- gate B (foreign owner refusal): pass. release skipped when GPU_LOCK_OWNER != held value, foreign lock intact
- gate C (contended backoff): pass. rc=1 after 63s (2+4+8+16+32=62s + overhead), error msg includes last-holder
- gate D (SIGTERM trap): pass. lock held confirmed pre-kill, EXISTS=0 post-SIGTERM via EXIT trap
- gate E (bash -n syntax): pass. all three scripts exit 0 (transcribe-headless.sh, batch.sh, docker/transcribe.sh)

**Why:** qdrant query (step 1) could not execute — mcp**qdrant-shared-stdio** not wired in this project's .mcp.json. Persisted spec was not retrieved; gates executed from caller-supplied spec only.

**How to apply:** if a follow-up task retrieves the persisted qdrant spec and finds additional gates, re-run only those gates. all five caller-supplied gates are green.
