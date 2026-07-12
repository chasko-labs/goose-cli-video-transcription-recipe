---
name: subshell pid scoping for semaphore tests
description: acquire+release must run in the same bash process or PIDs differ and acquire fails on second call
type: feedback
---

When testing gpu_acquire + gpu_release, both calls must happen inside a single `bash -c '...'` invocation. Split calls (acquire in one bash -c, release in another) produce different PIDs — the second shell cannot acquire the lock set by the first, and GPU_LOCK_OWNER is not exported across process boundaries.

**Why:** GPU_LOCK_OWNER is an exported var in the acquiring process's environment. A new bash -c subprocess does not inherit it unless explicitly passed. The owner value also embeds `$$` (PID), so two separate subshells produce different owner strings.

**How to apply:** in all gate scripts that test the full acquire→release cycle, source the semaphore functions and run both calls in the same `bash -c` block. For trap-fire gates (Gate D), the subshell spawned with `( ) &` is the process that owns the lock — the PID shown in the owner string is that subgroup's leader PID, not the background job PID visible to the caller.
