---
name: fc-pool health endpoint
description: confirmed fc-pool healthz endpoint and port for preflight checks in transcribe-headless.sh
type: reference
---

fc-pool runs as a systemd unit on `localhost:8150`. the health endpoint is `/healthz` — not `/health`, not `/`.

verified 2026-04-20:

- `GET http://127.0.0.1:8150/healthz` — HTTP 200, body `ok`
- `GET http://127.0.0.1:8150/health` — HTTP 404
- `GET http://127.0.0.1:8150/` — HTTP 404

probe pattern used in transcribe-headless.sh batch preflight:

```bash
curl -sf --max-time 3 "${FC_POOL_URL}/healthz" >/dev/null 2>&1
```

`FC_POOL_URL` defaults to `http://localhost:8150` (line 56 of transcribe-headless.sh).
