---
name: fail-fast guards — patch batch and retry patterns
description: lessons from implementing circuit breaker, gpu probe, and timeout tightening on fix/fail-fast-guards
type: project
---

circuit breaker is sequential-only — parallel batch mode (--parallel N>1) does not
have deterministic failure ordering across concurrent pipelines so the 3-consecutive
count is meaningless. document this limit explicitly in code and to callers.

**Why:** tried to make it work for parallel mode, found that backgrounded subshells
can complete in any order — the "last 3 failures" counter would measure arbitrary
interleaving, not actual consecutive failures on the same error shape.

**How to apply:** any consecutive-failure detection on a shared counter only works
in sequential execution mode. flag this clearly whenever circuit breakers are added
to parallel batch systems.

gpu_health_probe uses WHISPER_IMAGE for both whisper and vision stages — both hit
the same HIP init path (torch.cuda.device_count) so one image suffices. this avoids
needing a third image or a dedicated probe image.

**Why:** vision image may not be on disk on first run (built lazily). whisper image
is always present before stage 1. probe runs before stage 2 which always follows
stage 1, so whisper image is guaranteed cached.

valkey SCAN for key cleanup pattern: SCAN cursor MATCH "pattern:\*" COUNT 100 in a
loop until cursor returns "0". used in \_circuit_reset to clean up batch-specific
keys without TTL.

**How to apply:** always pair "no-TTL" valkey keys with explicit cleanup. batch-scoped
keys are safe without TTL only because process_batch calls \_circuit_reset at end.
any path that can abort early (signal trap) should also call \_circuit_reset.

update_status failure_reason classification:
124 = timeout (timeout command killed docker client)
137 = SIGKILL (docker daemon OOM or external kill)
125 = docker exec itself failed (image missing, daemon down)
anything else = application-level exit code

**How to apply:** use this classification table any time pipeline exit codes need
to be bucketed into retry vs no-retry vs transient vs permanent.
