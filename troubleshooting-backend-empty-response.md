# Jarvis Backend - Empty Response on Port 3000

**Date:** 2026-06-11

## Issue

The `jarvis-backend-1` container is running and logs show "Server is running on port 3000", but all HTTP requests to port 3000 return an empty response (`net::ERR_EMPTY_RESPONSE`). The frontend at `:8080` keeps retrying `GET http://100.82.114.45:3000/health` and failing.

## Symptoms

- `curl -sv http://localhost:3000/health` — connects but receives "Empty reply from server"
- `curl -sv http://localhost:3000/` — same empty reply
- Frontend browser console shows repeated `ERR_EMPTY_RESPONSE` on `/health`
- STT WebSocket on port 9001 appears to work
- Logs show all models loaded successfully (Whisper, EOU, Kokoro TTS)

## Container Status at Time of Issue

| Container | Port Mapping | Status |
|-----------|-------------|--------|
| jarvis-backend-1 | 3000->3000, 9001->9001 | Up, but HTTP not responding |
| jarvis-frontend-1 | 443->443, 8080->80 | Up, working |
| jarvis-qdrant-1 | 6333->6333 | Up, working |

## Working Endpoints

- Frontend: http://localhost:8080 or https://localhost:443
- Qdrant dashboard: http://localhost:6333/dashboard
- Qdrant API: http://localhost:6333/collections

## Backend Logs (last output)

```
STT: [STT] Loading faster-whisper model: medium.en on cuda (float16)
STT: [STT] Model loaded
STT: [EOU] EOU model loaded (threshold: 0.011)
Python script is ready
[STT] Starting STT WebSocket server on port 9001
Streaming STT server is ready
Kokoro: Loading pipeline (lang=a, voice=af_heart)...
Kokoro: Model loaded successfully
Streaming STT server started (provider: local)
Server is running on port 3000
```

## GPU Warning (non-critical)

```
GPU device discovery failed: device_discovery.cc:91 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
```

## Possible Causes

1. Backend HTTP server process crashed or hung after startup
2. Port 3000 bound by a subprocess (STT/model loader) instead of the main API server
3. Application-level error not surfacing in stdout/stderr logs
4. IPv6 binding issue — curl connects via `::1` (IPv6) while server may be IPv4-only

## Potential Fixes

1. Restart backend container: `docker restart jarvis-backend-1`
2. Check if the backend app binds to `0.0.0.0` vs `127.0.0.1`
3. Inspect backend app code for the port 3000 listener initialization
4. Try curl over IPv4 explicitly: `curl -4 http://127.0.0.1:3000/health`
5. Check `docker logs jarvis-backend-1` after restart for any new errors
6. Run `docker exec -it jarvis-backend-1 bash` and test health endpoint from inside the container