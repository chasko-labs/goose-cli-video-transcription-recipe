#!/usr/bin/env bash
# rewrite 2026-04-20 — bash is thin glue, logic lives in fc-pool /transcribe/*
# manifest iteration, status.json stamping, subshell throttle all owned by
# hs-transcribe batch handler (fc-pool /transcribe/batch endpoint).
# this script: arg parsing, env defaults, delegation, optional watch tail.
set -euo pipefail

# --- defaults ---
MANIFEST=""
MAX_PARALLEL=1
STAGE="whisper"
WATCH=false
EXTRA_ARGS=()

# --- arg parse ---
usage() {
  cat >&2 <<EOF
usage: batch.sh --manifest <path> [--max-parallel <n>] [--stage <str>] [--watch]
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --watch)
      WATCH=true
      shift
      ;;
    --help | -h)
      usage
      ;;
    *)
      # pass unknown flags through to hs-transcribe (e.g. --force, --dry-run)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "[batch] error: --manifest is required" >&2
  usage
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "[batch] error: manifest not found: $MANIFEST" >&2
  exit 1
fi

# --- env ---
FC_POOL_URL="${FC_POOL_URL:-http://localhost:8150}"
HS_TRANSCRIBE="${HS_TRANSCRIBE:-hs-transcribe}"

# --- fc-pool reachability preflight ---
# transport boundary:
#   endpoint: GET ${FC_POOL_URL}/healthz
#   timeout:  5s — fc-pool is loopback; any delay beyond 1s indicates daemon issue
#   on unreachable: exit 1; batch runs are supervised and should not silently degrade
if ! curl -sf --max-time 5 "${FC_POOL_URL}/healthz" >/dev/null 2>&1; then
  echo "[batch] error: fc-pool unreachable at ${FC_POOL_URL}/healthz" >&2
  echo "[batch] start fc-pool (systemd unit on port 8150) then retry" >&2
  exit 1
fi

# --- delegate to hs-transcribe batch ---
# hs-transcribe owns: manifest iteration, status.json stamping, wait-n throttle,
# gpu_lock serialization across parallel slots, retry classification, jaeger traces.
# --watch: if set, poll hs-transcribe status <job-id> --watch after batch submit
#          and tail its output to tty.
echo "[batch] dispatching to ${HS_TRANSCRIBE} batch" >&2

if [[ "$WATCH" == true ]]; then
  # extract job_id from --json response (hs-transcribe has no --print-job-id flag)
  JOB_ID=$(
    "$HS_TRANSCRIBE" --json batch \
      --manifest "$MANIFEST" \
      --max-parallel "$MAX_PARALLEL" \
      --stage "$STAGE" \
      "${EXTRA_ARGS[@]}" |
      jq -r '.job_id'
  )
  if [[ -z "$JOB_ID" || "$JOB_ID" == "null" ]]; then
    echo "[batch] failed to parse job_id from hs-transcribe batch response" >&2
    exit 2
  fi
  echo "[batch] job-id: ${JOB_ID}" >&2
  exec "$HS_TRANSCRIBE" status "$JOB_ID" --watch
else
  exec "$HS_TRANSCRIBE" batch \
    --manifest "$MANIFEST" \
    --max-parallel "$MAX_PARALLEL" \
    --stage "$STAGE" \
    "${EXTRA_ARGS[@]}"
fi
