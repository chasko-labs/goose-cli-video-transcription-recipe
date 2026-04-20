#!/bin/bash
# transcribe-headless.sh — full pipeline: media-extract -> whisper -> vision -> merge -> narrative
# called via SSH from mac mini:
#   ssh rocm-aibox "bash ~/code/projects/goose-cli-video-transcription-recipe/scripts/transcribe-headless.sh <url> [model]"
#
# flags:
#   --force           re-run all stages even if outputs exist
#   --dry-run         print plan without executing
#   --batch <file>    process URLs from file (one per line, # comments ok)
#   --parallel <N>    concurrent pipelines in batch mode (default: 1)
set -uo pipefail

# --- argument parsing ---
FORCE=false
DRY_RUN=false
BATCH_FILE=""
PARALLEL_N=1
AUDIO_ONLY=auto
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      FORCE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --batch)
      BATCH_FILE="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL_N="$2"
      shift 2
      ;;
    --audio-only)
      AUDIO_ONLY=true
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

# --- config ---
PROJECT_DIR="$HOME/code/projects/goose-cli-video-transcription-recipe"
DATA_DIR="${VIDEO_TRANSCRIPTS_DIR:-$HOME/video-transcripts}"
WHISPER_IMAGE="goose-cli-video-transcription-recipe-whisper:latest"
VISION_IMAGE="goose-cli-video-transcription-recipe-vision:latest"
NARRATIVES_DIR="$DATA_DIR/narratives"
FC_POOL_URL="${FC_POOL_URL:-http://localhost:8150}"
JAEGER_OTLP="${JAEGER_OTLP:-http://rocm-aibox.local:4318}"
TRACE_PY="$PROJECT_DIR/scripts/trace.py"

# stage timeouts (seconds, overridable via env)
TIMEOUT_STAGE0="${TIMEOUT_STAGE0:-600}"
TIMEOUT_STAGE1="${TIMEOUT_STAGE1:-900}"
TIMEOUT_STAGE2="${TIMEOUT_STAGE2:-600}"
TIMEOUT_STAGE3="${TIMEOUT_STAGE3:-60}"
TIMEOUT_STAGE4="${TIMEOUT_STAGE4:-180}"

STAGE_NAMES=("media-extract" "whisper" "vision" "merge" "narrative")

# container writes land on host bind mounts — run as host uid:gid so files are
# owned by $USER:heraldstack instead of root. heraldstack group is shared across
# human + container identities. override via DOCKER_USER_ARGS if needed.
_HS_GID=$(getent group heraldstack 2>/dev/null | cut -d: -f3)
if [[ -n "$_HS_GID" ]]; then
  DOCKER_USER_ARGS="${DOCKER_USER_ARGS:---user $(id -u):${_HS_GID} --group-add ${_HS_GID}}"
else
  DOCKER_USER_ARGS="${DOCKER_USER_ARGS:-}"
fi

# --- helpers ---

now_s() { date +%s; }
now_ns() { date +%s%N 2>/dev/null || echo "$(date +%s)000000000"; }

update_status() {
  local stage="$1" stage_name="$2" status="$3" elapsed_s="${4:-}"
  python3 - "$VIDEO_DIR/status.json" "$URL" "$SLUG" "$stage" "$stage_name" "$status" "$elapsed_s" "$PIPELINE_STARTED_AT" <<'PYEOF'
import sys, json, os
path, url, slug, stage, stage_name, status, elapsed_s, started_at = sys.argv[1:9]
try:
    data = json.loads(open(path).read())
except Exception:
    data = {"url": url, "slug": slug, "stages": {}, "started_at": started_at}
data["stage"] = int(stage)
data["stage_name"] = stage_name
data["status"] = status
s = data.setdefault("stages", {}).setdefault(stage, {})
s["status"] = status
if elapsed_s:
    s["elapsed_s"] = int(elapsed_s)
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
os.rename(tmp, path)
PYEOF
}

send_trace() {
  local span_name="$1" start_ns="$2" end_ns="$3" status="$4"
  shift 4
  python3 "$TRACE_PY" send "$JAEGER_OTLP" "$TRACE_ID" \
    "$(python3 "$TRACE_PY" id "${span_name}-${SLUG}" 8)" \
    "$span_name" "$start_ns" "$end_ns" "$status" "$PIPELINE_SPAN" "$@" 2>/dev/null &
}

# --- ollama gpu eviction ---
#
# ollama keeps loaded models resident on the GPU for keep_alive seconds after
# last use (default 300s). on a 12GB AMD GPU this can hold ~7-8GB, leaving
# the vision container with 0 bytes free even though pytorch only needs ~4GB.
# observed: HIP OOM with 3.56 GiB allocated / 0 bytes free.
#
# fix: before whisper+vision stages, POST keep_alive=0 for every loaded model
# so ollama unloads them and returns GPU memory to the OS. narrative stage
# (stage 4) re-loads mistral-nemo / llama3.1:8b on demand — first call after
# eviction pays a cold-load cost, handled by OLLAMA_TIMEOUT=600 in
# generate-narrative.py.
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

ollama_evict() {
  local models
  models=$(curl -sf --max-time 5 "${OLLAMA_URL}/api/ps" 2>/dev/null |
    python3 -c 'import json,sys; d=json.load(sys.stdin); print("\n".join(m["name"] for m in d.get("models",[])))' 2>/dev/null)
  if [[ -z "$models" ]]; then
    return 0
  fi
  while IFS= read -r m; do
    [[ -z "$m" ]] && continue
    echo "[ollama-evict] unloading $m"
    curl -sf --max-time 10 -X POST "${OLLAMA_URL}/api/generate" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null 2>&1 || true
  done <<<"$models"
}

# --- gpu semaphore (t-02) ---
#
# transport boundary:
#   key:     gpu_lock
#   value:   <pid>-<hostname>-<stage>  (owner_id — used for compare-and-delete)
#   acquire: SET gpu_lock <owner_id> NX EX <ttl>
#            NX = only set if absent (mutex)
#            ttl = per-stage orphan backstop passed to gpu_acquire (default 10800).
#            stage 0 fallback (whisper+download) observed at 7170s; vision at ~900s.
#            DEL on clean exit is load-bearing; TTL is last-resort only.
#   release: GET gpu_lock → compare to owner_id → DEL only if match
#            prevents a ttl-expired-then-reacquired race from releasing a new
#            owner's lock. no lua required: a single pid can only own one stage
#            at a time and releases only what it acquired.
#   retry:   exponential backoff, base 2s, max 5 attempts (2,4,8,16,32 = 62s)
#            fail-fast after 5 misses with actionable error message
#   cleanup: trap EXIT/SIGINT/SIGTERM in every subshell that calls gpu_acquire
#            outer trap at run_pipeline() level covers tmpfile cleanup + lock release
#
# valkey transport: docker exec heraldstack-valkey valkey-cli
#   (no redis-cli / valkey-cli binary on the aibox host; valkey/valkey:latest
#    container at localhost:6379 exposes valkey-cli internally)
# env overrides: VALKEY_CONTAINER (default: heraldstack-valkey)
#                VALKEY_HOST / VALKEY_PORT (informational only for this impl;
#                docker exec ignores them — change valkey_cmd() if host cli
#                becomes available)

VALKEY_CONTAINER="${VALKEY_CONTAINER:-heraldstack-valkey}"

# low-level valkey command wrapper — all semaphore calls go through here
valkey_cmd() {
  docker exec "$VALKEY_CONTAINER" valkey-cli "$@" 2>/dev/null
}

# gpu_acquire <stage> [ttl [max_wait]]
#   sets gpu_lock to "<pid>-<hostname>-<stage>" with NX EX <ttl>
#   ttl defaults to 10800 (3h) — callers should pass a stage-appropriate value
#   max_wait: total seconds to keep retrying (default 3600); exponential backoff capped at 60s
#   on success: exports GPU_LOCK_OWNER for use by gpu_release
#   on failure: prints actionable error, returns 1
gpu_acquire() {
  local stage="$1"
  local ttl="${2:-10800}"
  local max_wait="${3:-3600}"
  local owner
  owner="$$-$(hostname -s)-${stage}"
  local delay=2
  local attempt=0
  local elapsed=0

  local my_host
  my_host=$(hostname -s)
  while ((elapsed < max_wait)); do
    local result
    result=$(valkey_cmd SET gpu_lock "$owner" NX EX "$ttl")
    if [[ "$result" == "OK" ]]; then
      echo "[gpu-lock] acquired: owner=$owner"
      GPU_LOCK_OWNER="$owner"
      export GPU_LOCK_OWNER
      return 0
    fi
    local held_by
    held_by=$(valkey_cmd GET gpu_lock 2>/dev/null || echo "unknown")
    # self-heal: holder is <pid>-<host>-<stage>. if host matches and pid is dead,
    # the lock is orphaned — delete and retry immediately. safer than waiting for ttl.
    local h_pid="${held_by%%-*}"
    local h_rest="${held_by#*-}"
    local h_host="${h_rest%%-*}"
    if [[ "$h_host" == "$my_host" && "$h_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$h_pid" 2>/dev/null; then
      echo "[gpu-lock] stale lock detected: holder pid $h_pid on $h_host is dead — releasing"
      valkey_cmd DEL gpu_lock >/dev/null
      continue
    fi
    echo "[gpu-lock] lock held by '$held_by' — retry $((attempt + 1)) in ${delay}s (${elapsed}/${max_wait}s elapsed)"
    sleep "$delay"
    elapsed=$((elapsed + delay))
    delay=$((delay * 2 > 60 ? 60 : delay * 2))
    ((attempt++))
  done

  echo "[gpu-lock] error: could not acquire gpu_lock after ${max_wait}s" \
    "(last holder: $(valkey_cmd GET gpu_lock 2>/dev/null || echo 'unknown'))" >&2
  return 1
}

# gpu_release
#   GET gpu_lock, compare to GPU_LOCK_OWNER, DEL only if match
#   safe to call when no lock is held (no-op)
gpu_release() {
  local owner="${GPU_LOCK_OWNER:-}"
  if [[ -z "$owner" ]]; then
    return 0
  fi
  local held_by
  held_by=$(valkey_cmd GET gpu_lock 2>/dev/null || echo "")
  if [[ "$held_by" == "$owner" ]]; then
    valkey_cmd DEL gpu_lock >/dev/null
    echo "[gpu-lock] released: owner=$owner"
  else
    echo "[gpu-lock] release skipped: lock not owned by us (held='$held_by', ours='$owner')"
  fi
  GPU_LOCK_OWNER=""
}

# --- dry-run ---

do_dry_run() {
  local url="$1" model="$2"
  local slug
  slug=$(echo "$url" | sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60)
  local vdir="$DATA_DIR/$slug"

  echo "--- dry-run ---"
  echo "url:        $url"
  echo "slug:       $slug"
  echo "model:      $model"
  echo "audio-only: $AUDIO_ONLY"
  echo "dir:        $vdir"
  echo ""

  # docker images
  for img in "$WHISPER_IMAGE" "$VISION_IMAGE"; do
    if docker image inspect "$img" >/dev/null 2>&1; then
      echo "image: $img (exists)"
    else
      echo "image: $img (NEEDS BUILD)"
    fi
  done

  # fc-pool reachability
  if curl -sf --max-time 3 "${FC_POOL_URL}/" >/dev/null 2>&1; then
    echo "fc-pool: reachable at $FC_POOL_URL"
  else
    echo "fc-pool: UNREACHABLE at $FC_POOL_URL (will use fallback)"
  fi
  echo ""

  # stage skip check
  local base
  base=$(ls -t "${vdir}/audio/"*.wav 2>/dev/null | head -1 | xargs -I{} basename {} .wav 2>/dev/null)
  local stages=("audio/*.wav" "transcripts/${base}.json" "transcripts/${base}-frame-analysis.json" "transcripts/${base}-combined.json")
  for i in 0 1 2 3; do
    local pattern="${stages[$i]}"
    local found
    found=$(ls ${vdir}/${pattern} 2>/dev/null | head -1)
    if [[ -n "$found" ]] && [[ "$FORCE" = false ]]; then
      echo "stage $i (${STAGE_NAMES[$i]}): SKIP (output exists)"
    else
      echo "stage $i (${STAGE_NAMES[$i]}): RUN"
    fi
  done
  # stage 4: check narratives
  if ls "$NARRATIVES_DIR/"*.md >/dev/null 2>&1 && [[ "$FORCE" = false ]]; then
    echo "stage 4 (${STAGE_NAMES[4]}): SKIP (narrative exists — may re-run if title differs)"
  else
    echo "stage 4 (${STAGE_NAMES[4]}): RUN"
  fi
}

# --- single video pipeline ---

run_pipeline() {
  local URL="$1"
  local MODEL="$2"

  SLUG=$(echo "$URL" | sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60)
  VIDEO_DIR="$DATA_DIR/$SLUG"
  PIPELINE_STARTED_AT=$(date -Iseconds)
  local PIPELINE_START
  PIPELINE_START=$(now_s)
  local PIPELINE_START_NS
  PIPELINE_START_NS=$(now_ns)

  # tracing IDs
  TRACE_ID=$(python3 "$TRACE_PY" id "${SLUG}-${PIPELINE_STARTED_AT}" 16)
  PIPELINE_SPAN=$(python3 "$TRACE_PY" id "pipeline-${SLUG}" 8)

  cd "$PROJECT_DIR"
  mkdir -p "$VIDEO_DIR"/{videos,audio,frames,transcripts} "$NARRATIVES_DIR"

  # stamp whisper_model into status.json so downstream tooling (route-narrative)
  # can read it from a single source of truth
  python3 - "$VIDEO_DIR/status.json" "$MODEL" <<'PYEOF'
import sys, json, os
path, model = sys.argv[1], sys.argv[2]
try:
    data = json.loads(open(path).read())
except Exception:
    data = {}
data["whisper_model"] = model
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
os.rename(tmp, path)
PYEOF

  # build images if missing
  if ! docker image inspect "$WHISPER_IMAGE" >/dev/null 2>&1; then
    echo "[pipeline] building whisper container (first run)"
    docker build -f docker/whisper-rocm.Dockerfile -t "$WHISPER_IMAGE" docker/
  fi
  if ! docker image inspect "$VISION_IMAGE" >/dev/null 2>&1; then
    echo "[pipeline] building vision container (first run)"
    docker build -f docker/vision-rocm.Dockerfile -t "$VISION_IMAGE" docker/
  fi

  echo "[pipeline] video dir: $VIDEO_DIR"
  local BASE=""
  local WHISPER_DONE=false

  # defensive defaults — guards against referencing unset vars if stages skip
  local S0_ELAPSED=0 S1_ELAPSED=0 S2_ELAPSED=0 S3_ELAPSED=0 S4_ELAPSED=0

  # pre-stage-0 audio-only detection from URL pattern
  if [[ "$AUDIO_ONLY" = "auto" ]]; then
    case "$URL" in
      *.mp3* | *.wav* | *.ogg* | *.m4a* | *.aac* | *.flac* | *podcast* | *audio.mp3*)
        AUDIO_ONLY=true
        echo "[pipeline] auto-detected audio-only (url pattern)"
        ;;
    esac
  fi

  # =========================================================================
  # stage 0: media-extract via fc-pool
  # =========================================================================
  local S0_START S0_START_NS S0_END
  S0_START=$(now_s)
  S0_START_NS=$(now_ns)

  if [[ "$FORCE" = false ]] && ls "${VIDEO_DIR}/audio/"*.wav >/dev/null 2>&1; then
    BASE=$(ls -t "${VIDEO_DIR}/audio/"*.wav | head -1 | xargs -I{} basename {} .wav)
    echo "[pipeline] stage 0: skipped (output exists)"
    S0_END=$(now_s)
    S0_ELAPSED=$((S0_END - S0_START))
    update_status 0 "media-extract" "done" "$S0_ELAPSED"
    # check if whisper output also exists (from prior fallback run)
    if [[ -f "$VIDEO_DIR/transcripts/${BASE}.json" ]]; then
      WHISPER_DONE=true
    fi
  elif [[ "$AUDIO_ONLY" = true ]]; then
    # audio-only stage 0: download + convert to wav, skip frames
    echo "[pipeline] stage 0/4: audio-only media extract"
    update_status 0 "media-extract" "running"

    BASE=$(date +%Y%m%d_%H%M%S)_$(echo "$SLUG" | cut -c1-40)
    # container name: deterministic per pid+stage so the kill trap can target it
    _S0A_CNAME="whisper-$$-s0audio"
    # cleanup contract: kill container on exit — covers timeout path where docker
    # run client is killed but daemon keeps container running indefinitely.
    trap 'docker kill "'"$_S0A_CNAME"'" 2>/dev/null; docker rm -f "'"$_S0A_CNAME"'" 2>/dev/null; rm -f "$S1_RESULT" "$S2_RESULT"; gpu_release' EXIT SIGINT SIGTERM
    # shellcheck disable=SC2086  # DOCKER_USER_ARGS intentionally word-splits into multiple flags
    timeout "$TIMEOUT_STAGE0" docker run --rm --name "$_S0A_CNAME" $DOCKER_USER_ARGS \
      -v "$VIDEO_DIR/audio:/media/audio" \
      -v "$VIDEO_DIR/videos:/media/videos" \
      -v "$VIDEO_DIR/transcripts:/media/transcripts" \
      --entrypoint bash \
      "$WHISPER_IMAGE" -c "
        set -e
        echo '[audio-extract] downloading audio'
        yt-dlp --no-playlist -o '/media/videos/${BASE}.%(ext)s' '${URL}' 2>&1
        SRC=\$(ls -t /media/videos/${BASE}.* 2>/dev/null | head -1)
        if [ -z \"\$SRC\" ]; then
          echo '[audio-extract] error: no file downloaded' >&2
          exit 1
        fi
        echo \"[audio-extract] converting \$SRC to wav\"
        ffmpeg -i \"\$SRC\" -ar 16000 -ac 1 -c:a pcm_s16le \"/media/audio/${BASE}.wav\" -y 2>&1
        echo '[audio-extract] fetching metadata'
        yt-dlp --dump-json --no-download '${URL}' 2>/dev/null | grep '^{' > /media/transcripts/metadata.json || true
        echo '[audio-extract] done'
      " 2>&1

    if [[ ! -f "$VIDEO_DIR/audio/${BASE}.wav" ]]; then
      echo "[pipeline] error: audio-only extract failed" >&2
      S0_END=$(now_s)
      S0_ELAPSED=$((S0_END - S0_START))
      update_status 0 "media-extract" "failed" "$S0_ELAPSED"
      send_trace "stage-0-media-extract" "$S0_START_NS" "$(now_ns)" "error" "stage=0" "mode=audio-only"
      return 1
    fi
  else
    echo "[pipeline] stage 0/4: media-extract (fc-pool)"
    update_status 0 "media-extract" "running"

    MEDIA_EXTRACT_BODY=$(
      cat <<JSONEOF
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"extract_media","arguments":{"url":"${URL}","model":"${MODEL}"}}}
JSONEOF
    )

    FC_RESULT=$(timeout "$TIMEOUT_STAGE0" curl -sf \
      --max-time "$TIMEOUT_STAGE0" \
      -H "Content-Type: application/json" \
      -d "$MEDIA_EXTRACT_BODY" \
      "${FC_POOL_URL}/call/media-extract?data_dir=${VIDEO_DIR}" 2>&1) && FC_OK=true || FC_OK=false

    if [ "$FC_OK" = true ]; then
      echo "[pipeline] media-extract via fc-pool done"
      BASE=$(echo "$FC_RESULT" | python3 -c "
import sys, json
try:
    resp = json.load(sys.stdin)
    content = resp.get('result', {}).get('content', [])
    for item in content:
        if item.get('type') == 'text':
            data = json.loads(item['text'])
            print(data.get('base', ''))
            break
except Exception:
    pass
" 2>/dev/null)
      if [ -z "$BASE" ]; then
        echo "[pipeline] warning: could not parse base from fc-pool response; checking video dir"
        BASE=$(ls -t "${VIDEO_DIR}/audio/"*.wav 2>/dev/null | head -1 | xargs -I{} basename {} .wav)
      fi
    fi

    if [ "$FC_OK" = false ] || [ -z "$BASE" ]; then
      echo "[pipeline] fc-pool unavailable or parse failed — falling back to in-container media extract"
      # transport boundary: acquire gpu_lock before the fallback docker-run.
      # this path runs in the main shell (not a subshell), so the outer EXIT trap
      # on run_pipeline() covers release if the function returns early or the
      # process is killed. gpu_release is idempotent when GPU_LOCK_OWNER is unset.
      if ! gpu_acquire "fallback" 10800; then
        echo "[pipeline] error: could not acquire gpu_lock for stage-0 fallback" >&2
        S0_END=$(now_s)
        S0_ELAPSED=$((S0_END - S0_START))
        update_status 0 "media-extract" "failed" "$S0_ELAPSED"
        send_trace "stage-0-media-extract" "$S0_START_NS" "$(now_ns)" "error" "stage=0"
        return 1
      fi
      # stage-0 fallback: stdout must be captured into WHISPER_OUT for base-prefix
      # parsing below, so we cannot use run_with_timeout (nested subshell loses $?).
      # inline pattern: name the container, write output to tmpfile, kill on EXIT.
      #
      # container name: deterministic per pid+stage
      # cleanup contract: docker kill fires on EXIT (timeout or signal) — daemon
      # keeps container alive after client kill without this trap.
      _S0F_CNAME="whisper-$$-s0fallback"
      _S0F_OUT=$(mktemp)
      trap 'docker kill "'"$_S0F_CNAME"'" 2>/dev/null; docker rm -f "'"$_S0F_CNAME"'" 2>/dev/null; rm -f "$_S0F_OUT" "$S1_RESULT" "$S2_RESULT"; gpu_release' EXIT SIGINT SIGTERM
      # shellcheck disable=SC2086  # DOCKER_USER_ARGS intentionally word-splits into multiple flags
      timeout "$TIMEOUT_STAGE0" docker run --rm --name "$_S0F_CNAME" $DOCKER_USER_ARGS \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --group-add render \
        -v "$VIDEO_DIR/videos:/media/videos" \
        -v "$VIDEO_DIR/audio:/media/audio" \
        -v "$VIDEO_DIR/frames:/media/frames" \
        -v "$VIDEO_DIR/transcripts:/media/transcripts" \
        -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
        -e PYTORCH_ALLOC_CONF=expandable_segments:True \
        -e SCENE_THRESHOLD=0.4 \
        "$WHISPER_IMAGE" "$URL" "$MODEL" >"$_S0F_OUT" 2>&1
      WHISPER_OUT=$(cat "$_S0F_OUT")
      rm -f "$_S0F_OUT"
      # release immediately after docker-run completes — stages 1+2 need the lock
      gpu_release
      echo "$WHISPER_OUT"

      BASE=$(echo "$WHISPER_OUT" | grep '\[transcribe\] transcript:' | sed 's|.*transcripts/||;s|\.\*||' | tr -d ' ')
      if [ -z "$BASE" ]; then
        echo "[pipeline] error: could not determine base prefix from whisper output" >&2
        S0_END=$(now_s)
        S0_ELAPSED=$((S0_END - S0_START))
        update_status 0 "media-extract" "failed" "$S0_ELAPSED"
        send_trace "stage-0-media-extract" "$S0_START_NS" "$(now_ns)" "error" "stage=0"
        return 1
      fi
      WHISPER_DONE=true
    fi

    S0_END=$(now_s)
    S0_ELAPSED=$((S0_END - S0_START))
    echo "[pipeline] stage 0 done (${S0_ELAPSED}s)"
    update_status 0 "media-extract" "done" "$S0_ELAPSED"
    send_trace "stage-0-media-extract" "$S0_START_NS" "$(now_ns)" "ok" "stage=0"

    # when fallback path ran whisper inside stage 0, record stage 1 as done
    if [[ "$WHISPER_DONE" = true ]]; then
      update_status 1 "whisper" "done" "0"
    fi
  fi

  echo "[pipeline] base: $BASE"

  # =========================================================================
  # audio-only inference
  # =========================================================================
  if [[ "$AUDIO_ONLY" = "auto" ]]; then
    # check frame count
    local FRAME_COUNT
    FRAME_COUNT=$(ls "${VIDEO_DIR}/frames/"*.png 2>/dev/null | wc -l)
    if [[ "$FRAME_COUNT" -eq 0 ]]; then
      AUDIO_ONLY=true
      echo "[pipeline] auto-detected audio-only (no frames extracted)"
    fi
    # check metadata for audio-only content
    if [[ "$AUDIO_ONLY" = "auto" ]] && [[ -f "$VIDEO_DIR/transcripts/metadata.json" ]]; then
      local VCODEC
      VCODEC=$(python3 -c "import json; m=json.load(open('$VIDEO_DIR/transcripts/metadata.json')); print(m.get('vcodec',''))" 2>/dev/null)
      if [[ "$VCODEC" == "none" ]]; then
        AUDIO_ONLY=true
        echo "[pipeline] auto-detected audio-only (vcodec=none)"
      fi
      # check categories for podcast/music
      local IS_PODCAST
      IS_PODCAST=$(python3 -c "
import json
m=json.load(open('$VIDEO_DIR/transcripts/metadata.json'))
cats=[c.lower() for c in m.get('categories',[])]
tags=[t.lower() for t in m.get('tags',[])]
print('true' if any(k in ' '.join(cats+tags) for k in ['podcast','audiobook','audio only']) else 'false')
" 2>/dev/null)
      if [[ "$IS_PODCAST" == "true" ]]; then
        AUDIO_ONLY=true
        echo "[pipeline] auto-detected audio-only (podcast/audio content)"
      fi
    fi
    # if still auto, default to false (video content)
    if [[ "$AUDIO_ONLY" = "auto" ]]; then
      AUDIO_ONLY=false
    fi
  fi

  # if audio-only, create stub frame-analysis so merge can proceed
  if [[ "$AUDIO_ONLY" = true ]] && [[ ! -f "$VIDEO_DIR/transcripts/${BASE}-frame-analysis.json" ]]; then
    python3 -c "
import json
with open('$VIDEO_DIR/transcripts/${BASE}-frame-analysis.json', 'w') as f:
    json.dump({'base': '$BASE', 'model': 'audio-only', 'frames': []}, f, indent=2)
"
    echo "[pipeline] audio-only: created stub frame-analysis"
  fi

  # =========================================================================
  # stages 1+2: whisper + vision (parallel GPU stages)
  # =========================================================================
  local S1_SKIP=false S2_SKIP=false

  # resume: check if outputs already exist
  if [[ "$FORCE" = false ]] && [[ -f "$VIDEO_DIR/transcripts/${BASE}.json" ]]; then
    S1_SKIP=true
  fi
  if [[ "$WHISPER_DONE" = true ]]; then
    S1_SKIP=true
  fi
  # skip vision if audio-only or output exists
  if [[ "$AUDIO_ONLY" = true ]]; then
    S2_SKIP=true
  fi
  if [[ "$FORCE" = false ]] && [[ -f "$VIDEO_DIR/transcripts/${BASE}-frame-analysis.json" ]]; then
    S2_SKIP=true
  fi

  local S1_RESULT S2_RESULT
  S1_RESULT=$(mktemp)
  S2_RESULT=$(mktemp)
  # trap covers both tmpfile cleanup and any gpu lock held by this pipeline invocation.
  # gpu_release is a no-op when GPU_LOCK_OWNER is unset, so safe to call unconditionally.
  # subshell traps (below) cover locks acquired inside backgrounded stages.
  trap "rm -f '$S1_RESULT' '$S2_RESULT'; gpu_release" EXIT SIGINT SIGTERM

  # --- stage 1: whisper ---
  if [[ "$S1_SKIP" = true ]]; then
    echo "[pipeline] stage 1: skipped (output exists)"
    update_status 1 "whisper" "done" "0"
    echo "0 0 0 0" >"$S1_RESULT"
  else
    echo "[pipeline] stage 1/4: whisper transcription (model=$MODEL)"
    update_status 1 "whisper" "running"
    (
      # transport boundary: SET gpu_lock <owner> NX EX 3600 before docker-run
      # cleanup contract: initial trap releases gpu lock; expanded below to also
      # kill the named container — set after cname is declared so the string
      # expansion captures the correct value.
      # container name: deterministic per pid+stage — docker kill targets this
      # on timeout so daemon does not keep the container running indefinitely.
      _S1_CNAME="whisper-$$-s1"
      trap 'docker kill "'"$_S1_CNAME"'" 2>/dev/null; docker rm -f "'"$_S1_CNAME"'" 2>/dev/null; gpu_release' EXIT SIGINT SIGTERM
      if ! gpu_acquire "whisper" 3600; then
        echo "[pipeline] error: could not acquire gpu_lock for whisper stage" >&2
        sns=$(now_ns)
        echo "1 0 $sns $(now_ns)" >"$S1_RESULT"
        exit 1
      fi
      # free GPU memory held by any ollama models kept warm from prior narrative/cli use
      ollama_evict
      s=$(now_s)
      sns=$(now_ns)
      AUDIO_FILE=$(ls -t "${VIDEO_DIR}/audio/${BASE}"*.wav 2>/dev/null | head -1)
      if [ -z "$AUDIO_FILE" ]; then
        echo "[pipeline] error: no audio file found for base=$BASE" >&2
        echo "1 0 $sns $(now_ns)" >"$S1_RESULT"
        exit 1
      fi
      # shellcheck disable=SC2086  # DOCKER_USER_ARGS intentionally word-splits into multiple flags
      timeout "$TIMEOUT_STAGE1" docker run --rm --name "$_S1_CNAME" $DOCKER_USER_ARGS \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --group-add render \
        -v "$VIDEO_DIR/audio:/media/audio:ro" \
        -v "$VIDEO_DIR/transcripts:/media/transcripts" \
        -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
        -e PYTORCH_ALLOC_CONF=expandable_segments:True \
        --entrypoint whisper \
        "$WHISPER_IMAGE" \
        "/media/audio/${BASE}.wav" \
        --model "$MODEL" \
        --output_dir /media/transcripts \
        --output_format all \
        --verbose False \
        2>&1
      rc=$?
      e=$(now_s)
      # gpu_release + docker kill called by EXIT trap above after rc is written
      echo "$rc $((e - s)) $sns $(now_ns)" >"$S1_RESULT"
      exit $rc
    ) &
    local PID1=$!
  fi

  # --- stage 2: vision ---
  if [[ "$S2_SKIP" = true ]]; then
    echo "[pipeline] stage 2: skipped (output exists)"
    update_status 2 "vision" "done" "0"
    echo "0 0 0 0" >"$S2_RESULT"
  else
    echo "[pipeline] stage 2/4: vision (Instella-VL-1B)"
    update_status 2 "vision" "running"
    (
      # transport boundary: SET gpu_lock <owner> NX EX 3600 before docker-run
      # vision acquires after whisper releases — stages are backgrounded in parallel
      # but the lock serializes actual GPU use. whisper PID1 holds the lock first;
      # vision will backoff and retry up to 62s total before failing.
      # container name: deterministic per pid+stage — kill trap fires on timeout
      # so daemon does not keep the container running indefinitely after client kill.
      _S2_CNAME="whisper-$$-s2"
      trap 'docker kill "'"$_S2_CNAME"'" 2>/dev/null; docker rm -f "'"$_S2_CNAME"'" 2>/dev/null; gpu_release' EXIT SIGINT SIGTERM
      if ! gpu_acquire "vision" 3600; then
        echo "[pipeline] error: could not acquire gpu_lock for vision stage" >&2
        sns=$(now_ns)
        echo "1 0 $sns $(now_ns)" >"$S2_RESULT"
        exit 1
      fi
      # free GPU memory held by any ollama models kept warm from prior narrative/cli use
      ollama_evict
      s=$(now_s)
      sns=$(now_ns)
      # shellcheck disable=SC2086  # DOCKER_USER_ARGS intentionally word-splits into multiple flags
      timeout "$TIMEOUT_STAGE2" docker run --rm --name "$_S2_CNAME" $DOCKER_USER_ARGS \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --group-add render \
        -v "$VIDEO_DIR/frames:/media/frames" \
        -v "$VIDEO_DIR/transcripts:/media/transcripts" \
        -v goose-cli-video-transcription-recipe_hf-cache:/cache/huggingface \
        -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
        -e PYTORCH_ALLOC_CONF=expandable_segments:True \
        -e VISION_MODEL=amd/Instella-VL-1B \
        "$VISION_IMAGE" /media/frames "$BASE" /media/transcripts \
        2>&1
      rc=$?
      e=$(now_s)
      # gpu_release + docker kill called by EXIT trap above after rc is written
      echo "$rc $((e - s)) $sns $(now_ns)" >"$S2_RESULT"
      exit $rc
    ) &
    local PID2=$!
  fi

  # wait for parallel stages
  local FAIL=0

  if [[ "$S1_SKIP" = false ]]; then
    wait "$PID1" || FAIL=1
  fi
  local S1_RC S1_ELAPSED S1_SNS S1_ENS
  read -r S1_RC S1_ELAPSED S1_SNS S1_ENS <"$S1_RESULT" 2>/dev/null || {
    S1_RC=1
    S1_ELAPSED=0
    S1_SNS=0
    S1_ENS=0
  }

  if [[ "$S1_SKIP" = false ]]; then
    if [[ "$S1_RC" == "0" ]]; then
      echo "[pipeline] stage 1 done (${S1_ELAPSED}s)"
      update_status 1 "whisper" "done" "$S1_ELAPSED"
      send_trace "stage-1-whisper" "$S1_SNS" "$S1_ENS" "ok" "stage=1" "model=$MODEL"
    else
      echo "[pipeline] stage 1 failed" >&2
      update_status 1 "whisper" "failed" "$S1_ELAPSED"
      send_trace "stage-1-whisper" "$S1_SNS" "$S1_ENS" "error" "stage=1"
    fi
  fi

  if [[ "$S2_SKIP" = false ]]; then
    wait "$PID2" || FAIL=1
  fi
  local S2_RC S2_ELAPSED S2_SNS S2_ENS
  read -r S2_RC S2_ELAPSED S2_SNS S2_ENS <"$S2_RESULT" 2>/dev/null || {
    S2_RC=1
    S2_ELAPSED=0
    S2_SNS=0
    S2_ENS=0
  }

  if [[ "$S2_SKIP" = false ]]; then
    if [[ "$S2_RC" == "0" ]]; then
      echo "[pipeline] stage 2 done (${S2_ELAPSED}s)"
      update_status 2 "vision" "done" "$S2_ELAPSED"
      send_trace "stage-2-vision" "$S2_SNS" "$S2_ENS" "ok" "stage=2"
    else
      echo "[pipeline] stage 2 failed" >&2
      update_status 2 "vision" "failed" "$S2_ELAPSED"
      send_trace "stage-2-vision" "$S2_SNS" "$S2_ENS" "error" "stage=2"
    fi
  fi

  rm -f "$S1_RESULT" "$S2_RESULT"

  if [[ $FAIL -ne 0 ]]; then
    update_status 2 "vision" "failed"
    return 1
  fi

  # =========================================================================
  # tighten transcript (between stages 1+2 and merge)
  # =========================================================================
  if [[ -f "$VIDEO_DIR/transcripts/${BASE}.json" ]]; then
    if [[ "$FORCE" = true ]] || [[ ! -f "$VIDEO_DIR/transcripts/${BASE}-tightened.json" ]]; then
      echo "[pipeline] tightening transcript"
      local TIGHT_START TIGHT_START_NS
      TIGHT_START=$(now_s)
      TIGHT_START_NS=$(now_ns)
      python3 "$PROJECT_DIR/scripts/tighten.py" transcript \
        "$VIDEO_DIR/transcripts/${BASE}.json" \
        "$VIDEO_DIR/transcripts/${BASE}-tightened.json"
      local TIGHT_ELAPSED=$(($(now_s) - TIGHT_START))
      send_trace "tighten-transcript" "$TIGHT_START_NS" "$(now_ns)" "ok" "type=transcript"
    else
      echo "[pipeline] tighten transcript: skipped (output exists)"
    fi
  fi

  # =========================================================================
  # stage 3: merge
  # =========================================================================
  local S3_START S3_START_NS S3_END S3_ELAPSED

  if [[ "$FORCE" = false ]] && [[ -f "$VIDEO_DIR/transcripts/${BASE}-combined.json" ]]; then
    echo "[pipeline] stage 3: skipped (output exists)"
    update_status 3 "merge" "done" "0"
  else
    echo "[pipeline] stage 3/4: merge"
    S3_START=$(now_s)
    S3_START_NS=$(now_ns)
    update_status 3 "merge" "running"

    # container name: deterministic per pid+stage
    # cleanup contract: run_with_timeout installs docker kill trap in its subshell
    # — container is killed when timeout fires, not just the docker run client.
    _S3_CNAME="whisper-$$-s3"
    # shellcheck disable=SC2086  # DOCKER_USER_ARGS intentionally word-splits into multiple flags
    if ! run_with_timeout "$TIMEOUT_STAGE3" "$_S3_CNAME" \
      docker run --rm $DOCKER_USER_ARGS \
      -v "$VIDEO_DIR:/media" \
      -v "$PROJECT_DIR/scripts:/scripts" \
      --entrypoint python3 \
      "$WHISPER_IMAGE" /scripts/merge-outputs.py "$BASE" /media; then
      S3_END=$(now_s)
      S3_ELAPSED=$((S3_END - S3_START))
      echo "[pipeline] stage 3 failed" >&2
      update_status 3 "merge" "failed" "$S3_ELAPSED"
      send_trace "stage-3-merge" "$S3_START_NS" "$(now_ns)" "error" "stage=3"
      return 1
    fi

    S3_END=$(now_s)
    S3_ELAPSED=$((S3_END - S3_START))
    echo "[pipeline] stage 3 done (${S3_ELAPSED}s)"
    update_status 3 "merge" "done" "$S3_ELAPSED"
    send_trace "stage-3-merge" "$S3_START_NS" "$(now_ns)" "ok" "stage=3"
  fi

  # =========================================================================
  # stage 4: narrative
  # =========================================================================
  local S4_START S4_START_NS S4_END S4_ELAPSED

  # check if narrative already exists for this video (grep for combined source path)
  local NARRATIVE_EXISTS=false
  if [[ "$FORCE" = false ]]; then
    local combined_ref="${BASE}-combined"
    if grep -rl "$combined_ref" "$NARRATIVES_DIR/"*.md >/dev/null 2>&1; then
      NARRATIVE_EXISTS=true
    fi
  fi

  if [[ "$NARRATIVE_EXISTS" = true ]]; then
    echo "[pipeline] stage 4: skipped (narrative exists)"
    update_status 4 "narrative" "done" "0"
  else
    echo "[pipeline] stage 4/4: narrative"
    S4_START=$(now_s)
    S4_START_NS=$(now_ns)
    update_status 4 "narrative" "running"

    if ! timeout "$TIMEOUT_STAGE4" python3 "$PROJECT_DIR/scripts/generate-narrative.py" "$VIDEO_DIR" "$NARRATIVES_DIR"; then
      S4_END=$(now_s)
      S4_ELAPSED=$((S4_END - S4_START))
      echo "[pipeline] stage 4 failed" >&2
      update_status 4 "narrative" "failed" "$S4_ELAPSED"
      send_trace "stage-4-narrative" "$S4_START_NS" "$(now_ns)" "error" "stage=4"
      return 1
    fi

    S4_END=$(now_s)
    S4_ELAPSED=$((S4_END - S4_START))
    echo "[pipeline] stage 4 done (${S4_ELAPSED}s)"
    update_status 4 "narrative" "done" "$S4_ELAPSED"
    send_trace "stage-4-narrative" "$S4_START_NS" "$(now_ns)" "ok" "stage=4"
  fi

  # =========================================================================
  # tighten narrative (after stage 4)
  # =========================================================================
  # find the narrative file that was just written
  local NARR_FILE
  NARR_FILE=$(ls -t "$NARRATIVES_DIR/"*.md 2>/dev/null | head -1)
  if [[ -n "$NARR_FILE" ]]; then
    echo "[pipeline] tightening narrative"
    local TIGHT_N_START TIGHT_N_START_NS
    TIGHT_N_START=$(now_s)
    TIGHT_N_START_NS=$(now_ns)
    timeout 120 python3 "$PROJECT_DIR/scripts/tighten.py" narrative "$NARR_FILE" 2>&1 ||
      echo "[pipeline] narrative tightening failed (non-fatal)"
    local TIGHT_N_ELAPSED=$(($(now_s) - TIGHT_N_START))
    echo "[pipeline] tighten narrative done (${TIGHT_N_ELAPSED}s)"
    send_trace "tighten-narrative" "$TIGHT_N_START_NS" "$(now_ns)" "ok" "type=narrative"

    # route narrative to qdrant gander-knowledge (non-fatal)
    # note: 2>&1 merges stdout (structured jsonl log) with stderr (human warnings).
    # downstream consumers can filter for structured events via | grep '"event":"route_narrative'
    # or split to separate log files if desired: 1>stdout.log 2>stderr.log
    echo "[pipeline] routing narrative to qdrant gander-knowledge"
    timeout 120 python3 "$PROJECT_DIR/scripts/route-narrative.py" "$NARR_FILE" "$VIDEO_DIR" 2>&1 ||
      echo "[pipeline] narrative routing failed (non-fatal)"

    # drift-scan narrative against chasko-labs/dotfiles-ops (non-fatal).
    # emits one assistant-message envelope per non-empty line of the narrative
    # as a workaround for heraldstack-firecracker#68 (multi-line content inside
    # a single JSONL record silently misses findings). Remove the line-split
    # when #68 is fixed.
    if [ -x /usr/local/bin/hs-drift-scan ]; then
      echo "[pipeline] drift-scan narrative -> chasko-labs/dotfiles-ops"
      python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        print(json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': line}}))
" "$NARR_FILE" |
        timeout 60 /usr/local/bin/hs-drift-scan --target chasko-labs/dotfiles-ops >/dev/null 2>&1 ||
        echo "[pipeline] drift-scan narrative failed (non-fatal)"
    fi
  fi

  # =========================================================================
  # pipeline complete
  # =========================================================================
  local PIPELINE_END PIPELINE_ELAPSED
  PIPELINE_END=$(now_s)
  PIPELINE_ELAPSED=$((PIPELINE_END - PIPELINE_START))

  # send pipeline-level trace span
  send_trace "pipeline" "$PIPELINE_START_NS" "$(now_ns)" "ok" \
    "url=$URL" "slug=$SLUG" "model=$MODEL"

  echo ""
  echo "[pipeline] done — $VIDEO_DIR (${PIPELINE_ELAPSED}s total)"
  echo "[pipeline] combined:   $VIDEO_DIR/transcripts/${BASE}-combined.md"
  echo "[pipeline] narrative:  $NARRATIVES_DIR/ (see above for filename)"
  echo "[pipeline] status:     $VIDEO_DIR/status.json"
  echo "[pipeline] traces:     $JAEGER_OTLP -> http://rocm-aibox.local:16686"
  return 0
}

# --- batch mode (sourced from batch.sh) ---
source "$PROJECT_DIR/scripts/batch.sh"

# --- container-kill-on-timeout helper ---
#
# transport boundary:
#   run_with_timeout <timeout_s> <cname> docker run [args...]
#
#   assigns --name <cname> to the docker run invocation so the container has a
#   deterministic identity for the kill path. runs in a subshell so the trap
#   is scoped to this helper call — does not interfere with the caller's traps.
#
#   cleanup contract:
#     trap 'docker kill <cname> 2>/dev/null; docker rm -f <cname> 2>/dev/null' EXIT
#     fires on: normal exit (container already gone via --rm), timeout kill,
#     SIGINT, SIGTERM forwarded from the parent process.
#     docker kill on an already-exited container is a no-op (exit 1, suppressed).
#     docker rm -f cleans up if --rm did not fire (timeout path killed the client
#     before docker daemon processed the --rm flag).
#
#   stdout/stderr: passed through to caller via subshell stdout.
#   return code:   the docker run exit code (or timeout's 124 on expiry).
#
#   NOTE: do not use this helper when caller needs stdout captured into a variable
#   via VARNAME=$(run_with_timeout ...) — the nested subshell eats the outer $?.
#   for those sites, patch inline (see stage-0 fallback below).

run_with_timeout() {
  local timeout_s="$1"
  local cname="$2"
  shift 2
  # $@ is now the full docker run invocation (without --name, we inject it)
  # find the position of "docker" and inject --name immediately after "run"
  local args=("$@")
  local patched=()
  local run_seen=false
  for arg in "${args[@]}"; do
    patched+=("$arg")
    if [[ "$run_seen" = false && "$arg" == "run" ]]; then
      patched+=("--name" "$cname")
      run_seen=true
    fi
  done

  (
    # kill the named container on any exit from this subshell — covers both the
    # normal path (container exited, docker kill is a no-op) and the timeout
    # path (client process killed, container still running in daemon).
    trap 'docker kill "'"$cname"'" 2>/dev/null; docker rm -f "'"$cname"'" 2>/dev/null' EXIT
    timeout "$timeout_s" "${patched[@]}"
  )
}

# --- main ---

if [[ -n "$BATCH_FILE" ]]; then
  MODEL="${POSITIONAL[0]:-medium}"

  if [[ "$DRY_RUN" = true ]]; then
    while IFS= read -r line; do
      line="${line%%#*}"
      line="$(echo "$line" | xargs)"
      [[ -z "$line" ]] && continue
      do_dry_run "$line" "$MODEL"
      echo ""
    done <"$BATCH_FILE"
    exit 0
  fi

  # lockfile on batch manifest path — prevents concurrent invocations against
  # the same batch file from stacking three pipeline trees on top of each other.
  #
  # transport boundary:
  #   lock path: /tmp/transcribe-<basename-of-BATCH_FILE>.lock
  #   flock -n:  non-blocking — fails immediately if lock is held
  #   held by:   the process that owns the flock fd (shell holding exec 9>...)
  #   release:   automatic on process exit (fd close), no explicit DEL needed
  #   scope:     batch mode only — single-URL mode does not need serialization
  #
  # on contention: print the holding pid and exit non-zero so the caller gets a
  # clear signal rather than silently stacking work.
  _BATCH_LOCK="/tmp/transcribe-$(basename "$BATCH_FILE").lock"
  exec 9>"$_BATCH_LOCK"
  if ! flock -n 9; then
    _HOLDING_PID=$(cat "$_BATCH_LOCK" 2>/dev/null || echo "unknown")
    echo "[batch] error: batch manifest '$BATCH_FILE' is already being processed" >&2
    echo "[batch] lock held at $_BATCH_LOCK (check pid via: fuser $_BATCH_LOCK)" >&2
    exit 1
  fi
  # write our pid into the lock file so contenders can identify the holder
  echo "$$" >"$_BATCH_LOCK"

  process_batch "$BATCH_FILE" "$PARALLEL_N" "$MODEL"
  exit $?
fi

# single video mode
URL="${POSITIONAL[0]:?usage: transcribe-headless.sh <video-url> [model-size] [--force] [--dry-run] [--batch <file>] [--parallel N]}"
MODEL="${POSITIONAL[1]:-medium}"

if [[ "$DRY_RUN" = true ]]; then
  do_dry_run "$URL" "$MODEL"
  exit 0
fi

run_pipeline "$URL" "$MODEL"
exit $?
