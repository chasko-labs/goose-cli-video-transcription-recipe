#!/usr/bin/env bash
# rewrite 2026-04-20 — bash is thin glue, logic lives in fc-pool /transcribe/*
# all stateful concerns (gpu_lock, docker lifecycle, retry classification,
# status.json, manifest iteration) handled by hs-transcribe / fc-pool.
# this script: arg parsing, env defaults, fc-pool reachability guard, delegation.
set -euo pipefail

# --- defaults ---
STAGE="whisper"
OUTPUT_DIR="${VIDEO_TRANSCRIPTS_DIR:-$HOME/video-transcripts}"
TIMEOUT="" # if unset, hs-transcribe uses its own stage default
URL=""
EXTRA_ARGS=()

# --- arg parse ---
usage() {
  cat >&2 <<EOF
usage: transcribe-headless.sh --url <url> [--stage whisper|vision] [--output-dir <path>] [--timeout <secs>]
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
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

if [[ -z "$URL" ]]; then
  echo "[transcribe] error: --url is required" >&2
  usage
fi

# --- stage-default timeouts ---
# caller may override via --timeout; if not set, hs-transcribe uses its own default.
# documented here so the contract is visible at the call site.
#   whisper default: 900s  (longer — large model + long audio)
#   vision  default: 600s
if [[ -z "$TIMEOUT" ]]; then
  case "$STAGE" in
    whisper) TIMEOUT=900 ;;
    vision) TIMEOUT=600 ;;
    *) TIMEOUT=900 ;;
  esac
fi

# --- env ---
FC_POOL_URL="${FC_POOL_URL:-http://localhost:8150}"
HS_TRANSCRIBE="${HS_TRANSCRIBE:-hs-transcribe}"

# --- fc-pool reachability preflight ---
# fail fast with a clean error rather than a silent 30-min hang if fc-pool is down.
# transport boundary:
#   endpoint: GET ${FC_POOL_URL}/healthz
#   timeout:  5s — fc-pool is loopback; any delay beyond 1s indicates daemon issue
#   on unreachable: exit 1 with diagnostic message
if ! curl -sf --max-time 5 "${FC_POOL_URL}/healthz" >/dev/null 2>&1; then
  echo "[transcribe] error: fc-pool unreachable at ${FC_POOL_URL}/healthz" >&2
  echo "[transcribe] start fc-pool (systemd unit on port 8150) then retry" >&2
  exit 1
fi

# --- delegate to hs-transcribe ---
# hs-transcribe owns: gpu_lock acquire/release, docker lifecycle, retry classification,
# status.json writes, stage orchestration, jaeger traces.
# this script owns: env setup, preflight, exit code propagation.
# structured json responses from hs-transcribe pass through on stdout untouched.
# diagnostic/progress lines go to stderr via hs-transcribe itself.
echo "[transcribe] dispatching to ${HS_TRANSCRIBE}" >&2
exec "$HS_TRANSCRIBE" run \
  --url "$URL" \
  --stage "$STAGE" \
  --timeout "$TIMEOUT" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
