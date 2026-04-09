#!/usr/bin/env python3
"""
trace.py — send OTLP spans to jaeger via HTTP JSON
Usage:
  trace.py id <seed> <bytes>          — deterministic hex ID from seed
  trace.py send <endpoint> <trace_id> <span_id> <name> <start_ns> <end_ns> <status> [parent_span_id] [key=val ...]

status: ok | error
endpoint: e.g. http://rocm-aibox.local:4318
"""
import sys
import json
import hashlib
import urllib.request


def gen_id(seed, n_bytes):
    return hashlib.sha256(seed.encode()).hexdigest()[:n_bytes * 2]


def send_span(endpoint, trace_id, span_id, name, start_ns, end_ns,
              status_code, parent_span_id=None, attrs=None):
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "status": {"code": 1 if status_code == "ok" else 2},
        "attributes": [
            {"key": k, "value": {"stringValue": v}}
            for k, v in (attrs or {}).items()
        ],
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id

    payload = {
        "resourceSpans": [{
            "resource": {
                "attributes": [{
                    "key": "service.name",
                    "value": {"stringValue": "video-pipeline"}
                }]
            },
            "scopeSpans": [{
                "scope": {"name": "transcribe-headless"},
                "spans": [span]
            }]
        }]
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/traces",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        print(f"[trace] warning: send failed for {name}: {e}", file=sys.stderr)


def main():
    cmd = sys.argv[1]
    if cmd == "id":
        print(gen_id(sys.argv[2], int(sys.argv[3])))
    elif cmd == "send":
        endpoint = sys.argv[2]
        trace_id = sys.argv[3]
        span_id = sys.argv[4]
        name = sys.argv[5]
        start_ns = int(sys.argv[6])
        end_ns = int(sys.argv[7])
        status = sys.argv[8]
        rest = sys.argv[9:]
        parent = None
        if rest and "=" not in rest[0]:
            parent = rest.pop(0)
        attrs = {}
        for arg in rest:
            if "=" in arg:
                k, v = arg.split("=", 1)
                attrs[k] = v
        send_span(endpoint, trace_id, span_id, name, start_ns, end_ns,
                  status, parent, attrs)


if __name__ == "__main__":
    main()
