#!/bin/bash
# batch.sh — batch orchestration for transcribe-headless.sh
# sourced by transcribe-headless.sh, not called directly
# expects: run_pipeline(), now_s(), DATA_DIR, FORCE, NARRATIVES_DIR, STAGE_NAMES

process_batch() {
  local batch_file="$1"
  local max_parallel="$2"
  local model="$3"
  local ts
  ts=$(date +%Y%m%d_%H%M%S)
  local manifest="$DATA_DIR/batch-${ts}.json"
  local tmpdir
  tmpdir=$(mktemp -d)

  # read URLs
  local urls=()
  while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue
    urls+=("$line")
  done < "$batch_file"

  local total=${#urls[@]}
  echo "[batch] $total URLs, parallel=$max_parallel"

  if (( total == 0 )); then
    echo "[batch] no URLs found in $batch_file"
    return 1
  fi

  mkdir -p "$DATA_DIR"
  local pids=()
  local active=0

  for i in "${!urls[@]}"; do
    local url="${urls[$i]}"
    local slug
    slug=$(echo "$url" | sed 's|https\?://||;s|www\.||;s|[^a-zA-Z0-9]|_|g' | sed 's|_\+|_|g;s|^_\|_$||g' | cut -c1-60)

    if (( max_parallel > 1 )); then
      # throttle to max_parallel
      while (( active >= max_parallel )); do
        wait -n 2>/dev/null
        ((active--))
      done
      (
        local start end rc
        start=$(now_s)
        run_pipeline "$url" "$model" > "$tmpdir/${i}.log" 2>&1
        rc=$?
        end=$(now_s)
        local status_str="done"
        [[ $rc -ne 0 ]] && status_str="failed"
        # find narrative path
        local narr
        narr=$(grep -o '\[narrative\] wrote .*' "$tmpdir/${i}.log" 2>/dev/null | sed 's/\[narrative\] wrote //' | tail -1)
        python3 -c "
import json
json.dump({
    'url': '$url',
    'slug': '$slug',
    'status': '$status_str',
    'elapsed_s': $((end - start)),
    'narrative_path': '$narr'
}, open('$tmpdir/${i}.json', 'w'))
"
        exit $rc
      ) &
      pids+=($!)
      ((active++))
    else
      local start end rc
      start=$(now_s)
      run_pipeline "$url" "$model" 2>&1 | tee "$tmpdir/${i}.log"
      rc=${PIPESTATUS[0]}
      end=$(now_s)
      local status_str="done"
      [[ $rc -ne 0 ]] && status_str="failed"
      local narr
      narr=$(grep -o '\[narrative\] wrote .*' "$tmpdir/${i}.log" 2>/dev/null | sed 's/\[narrative\] wrote //' | tail -1)
      python3 -c "
import json
json.dump({
    'url': '$url',
    'slug': '$slug',
    'status': '$status_str',
    'elapsed_s': $((end - start)),
    'narrative_path': '$narr'
}, open('$tmpdir/${i}.json', 'w'))
"
    fi
  done

  # wait for all parallel jobs
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
  done

  # build manifest + summary
  python3 - "$tmpdir" "$manifest" "$total" <<'PYEOF'
import sys, json, glob, os
tmpdir, manifest_path, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
results = []
for i in range(total):
    p = os.path.join(tmpdir, f"{i}.json")
    if os.path.exists(p):
        results.append(json.loads(open(p).read()))
    else:
        results.append({"url": "unknown", "slug": "", "status": "failed", "elapsed_s": 0, "narrative_path": ""})
with open(manifest_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n{'url':60s} | {'status':6s} | time")
print("-" * 80)
for r in results:
    u = r["url"][:58]
    print(f"  {u:58s} | {r['status']:6s} | {r['elapsed_s']}s")
PYEOF

  echo ""
  echo "[batch] manifest: $manifest"
  rm -rf "$tmpdir"
}
