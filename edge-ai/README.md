# Janus Edge AI Service

Production-oriented Python edge orchestration service for a home assistant robot system. This service does not own low-level hardware. It accepts typed device events from the ESP32-S3 runtime, maintains per-device session context, interprets touch input semantically, builds constrained LLM prompts, validates every outbound response plan, and returns only high-level semantic intents to the device runtime.

## Current System Summary

- Firmware boundary:
  The ESP32-S3 firmware owns microphone capture, speaker playback, face rendering, touch sensing, motor control, and real-time safety state.
- Edge AI boundary:
  This Python service owns device sessions, touch interpretation, prompt assembly, LLM orchestration, response validation, fallback behavior, and high-level response planning.
- WebSocket transport role:
  The ESP connects as a client over WebSocket, sends typed runtime events, receives typed `ack`, `error`, and `ai_response_plan` messages, and reconnects cleanly when needed.
- Prompt system role:
  The prompt builder assembles system prompt, developer constraints, dynamic runtime context, strict output instructions, and JSON schema for the active LLM adapter.
- Validation and fallback role:
  Raw provider output is parsed into a strict schema, semantically validated, and replaced with a safe fallback plan when output is malformed or unsafe.
- Current placeholders:
  Real STT, real TTS synthesis dispatch, persistent session storage, external telemetry export, and stronger device authentication are still placeholder or phase-2 work.

## What The Service Does

- Accepts WebSocket connections from ESP devices
- Requires a `hello` message before any other device event
- Tracks one active connection per `device_id`
- Replaces stale or duplicate connections cleanly
- Maintains in-memory session state per connected device
- Interprets raw touch events into semantic touch meaning
- Builds structured LLM prompts from runtime context
- Validates every AI response plan before dispatch
- Returns only semantic robot plans such as spoken text, emotion, face expression, voice style, and high-level actions

## Architecture

- FastAPI app shell
- Hexagonal structure with ports and adapters
- Thin WebSocket route
- Conversation orchestration in `ConversationOrchestrator`
- Prompt assembly in `PromptBuilder`
- Structured JSON schema parsing for LLM output
- Semantic validation and safe fallback services
- Provider boundary through `LlmPort`, `SttPort`, `TtsPort`, `TelemetryPort`, and `DeviceSessionRepositoryPort`
- Wake boundary through `WakeDetectionService`, with STT-based production detection and an explicit dev fake only for tests.

## ESP Connectivity Contract

The ESP runtime connects as a WebSocket client to:

- `ws://<host>:<port>/ws/device`
- `wss://<host>:<port>/ws/device` when deployed behind TLS or a reverse proxy

Expected device behavior:

1. Open the WebSocket connection.
2. Send `hello` as the first message.
3. Keep using the same `device_id` for the lifetime of that connection.
4. Send `heartbeat` regularly. Any valid message also counts as activity.
5. Optionally send `session_start` and `session_end` to make the interaction lifecycle explicit.
6. Reconnect using the same `device_id` if the socket closes. The server will replace the older connection safely.

Minimal incoming `hello` example:

```json
{
  "message_type": "hello",
  "device_id": "janus-esp-01",
  "protocol_version": "1.0",
  "firmware_version": "0.2.0",
  "capabilities": ["audio_in", "audio_out", "touch", "face_display"]
}
```

Minimal incoming `heartbeat` example:

```json
{
  "message_type": "heartbeat",
  "device_id": "janus-esp-01",
  "sequence": 42
}
```

Outgoing `ack` example:

```json
{
  "message_type": "ack",
  "device_id": "janus-esp-01",
  "session_id": "8af7f654-2f07-4a40-a4d2-6271f98bd2f0",
  "ack_for": "touch_event",
  "accepted": true,
  "message": "touch_processed",
  "server_time": "2026-04-22T12:00:00Z"
}
```

Outgoing `error` example:

```json
{
  "message_type": "error",
  "device_id": "janus-esp-01",
  "code": "protocol.invalid_message",
  "message": "Incoming websocket message failed schema validation.",
  "retryable": false,
  "server_time": "2026-04-22T12:00:00Z"
}
```

Notes:

- The first message must be `hello`, otherwise the server returns an error and closes the connection.
- If `EDGE_AI_ALLOWED_DEVICE_IDS` is configured, only listed device IDs may connect.
- If the ESP reconnects with the same `device_id`, the older connection is closed and replaced.
- This edge service never sends low-level hardware instructions. It only sends semantic response plans.

## Environment Variables

These are the currently used runtime settings:

- `EDGE_AI_APP_NAME`
- `EDGE_AI_ENVIRONMENT`
- `EDGE_AI_LOG_LEVEL`
- `EDGE_AI_LOG_JSON`
- `EDGE_AI_SERVER_HOST`
- `EDGE_AI_SERVER_PORT`
- `EDGE_AI_PROXY_HEADERS`
- `EDGE_AI_FORWARDED_ALLOW_IPS`
- `EDGE_AI_ROBOT_NAME`
- `EDGE_AI_DEFAULT_LANGUAGE`
- `EDGE_AI_WEBSOCKET_PATH`
- `EDGE_AI_ALLOWED_DEVICE_IDS`
- `EDGE_AI_DEVICE_AUTH_TOKEN`
- `EDGE_AI_LLM_PROVIDER`
- `EDGE_AI_STT_PROVIDER`
- `EDGE_AI_TTS_PROVIDER`
- `EDGE_AI_WAKE_DETECTOR_PROVIDER`
- `EDGE_AI_GEMINI_API_KEY`
- `EDGE_AI_GEMINI_MODEL_ID`
- `EDGE_AI_REQUEST_TIMEOUT_SECONDS`
- `EDGE_AI_MAX_AUDIO_CHUNKS_PER_SESSION`
- `EDGE_AI_MAX_WAKE_CHUNKS_PER_INTERACTION`
- `EDGE_AI_MAX_WAKE_BASE64_CHARS_PER_INTERACTION`
- `EDGE_AI_DEBUG_STORE_RAW_WAKE_AUDIO`
- `EDGE_AI_SESSION_HISTORY_LIMIT`
- `EDGE_AI_WEBSOCKET_HELLO_TIMEOUT_SECONDS`
- `EDGE_AI_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS`
- `EDGE_AI_WEBSOCKET_CLOSE_TIMEOUT_SECONDS`
- `EDGE_AI_WEBSOCKET_MAX_PROTOCOL_ERRORS`
- `EDGE_AI_WEBSOCKET_MAX_MESSAGE_BYTES`
- `EDGE_AI_WEBSOCKET_PING_INTERVAL_SECONDS`
- `EDGE_AI_WEBSOCKET_PING_TIMEOUT_SECONDS`

Use `.env.example` as the baseline. Do not commit real secrets.
Create a local `.env` from `.env.example` before starting the service.

## Development Run

This service is currently safest as a single-process deployment because:

- the session repository is in-memory
- the connection manager is in-process

Recommended development startup:

1. Copy `.env.example` to `.env`.
2. Adjust values as needed for your device and provider.
3. Start the service:

```bash
cd edge-ai
python -m app.server
```

Alternative entrypoint after installation:

```bash
cd edge-ai
janus-edge-ai
```

Health endpoint:

```text
GET /health
GET /ready
GET /version
```

## Hey Janus Wake Flow

Firmware sends bounded `wake_audio_chunk` windows while idle. The local firmware
prefilter only reduces traffic; it does not confirm wake. This service confirms
the wake phrase with `WakeDetectionService` and sends `wake_detected` or
`wake_rejected`.

After `wake_detected`, firmware sends `greeting_request`. The service synthesizes
`Size nasıl yardımcı olabilirim?` through `TtsPort` and returns bounded
`audio_output_chunk` messages followed by `audio_output_end`.

Use `EDGE_AI_WAKE_DETECTOR_PROVIDER=stt` for production. The `dev_fake` provider
is deterministic test tooling and is not production wake-word recognition.

## Docker Deployment

Build the image:

```bash
cd edge-ai
docker build -t janus-edge-ai:latest .
```

Run the container:

```bash
docker run --rm \
  --name janus-edge-ai \
  --env-file .env \
  -p 8080:8080 \
  janus-edge-ai:latest
```

Or use Compose:

```bash
cd edge-ai
docker compose up -d --build
```

Container notes:

- Runs as a non-root user
- Uses environment-driven config only
- Exposes port `8080` by default
- Keeps Uvicorn on a single worker intentionally so in-memory device/session state stays consistent

Production mode recommendation:

- Set `EDGE_AI_ENVIRONMENT=production`
- Keep `EDGE_AI_LOG_JSON=true`
- Place the service behind a reverse proxy if exposing it beyond a private LAN
- Treat `.env` as a deployment secret file and never bake secrets into the image

## Reverse Proxy Note

For a VPS, mini PC, or gateway deployment, prefer:

- TLS termination at a reverse proxy such as Caddy or Nginx
- `wss://` between ESP and public endpoint when traffic leaves the private LAN
- proxying `/ws/device` directly to this FastAPI service

This service does not enable browser CORS middleware by default because the primary client is the ESP device, not a browser app.

## Logging And Observability

The service emits structured logs useful for production debugging. Important fields include:

- `device_id`
- `session_id`
- `event_type`
- `provider`
- `orchestrator_phase`
- `error_category`

Telemetry logging redacts common secret-shaped keys such as token, secret, authorization, and api_key fields.

## Production-Ready Now

- Single-process FastAPI service startup
- Docker image and Compose example
- Environment-driven configuration
- Structured JSON logging
- Explicit ESP connection contract
- Required `hello` handshake
- Per-device connection replacement on reconnect
- Receive timeout and stale connection cleanup
- Typed ack and error responses
- Safe fallback behavior on orchestration or provider failure

## Placeholder Or Not Yet Complete

- Real STT transcription provider
- Real TTS generation and playback artifact delivery
- Fully validated Gemini production rollout under real traffic
- Persistent session repository for multi-process or multi-instance deployment
- External metrics or tracing backend
- Strong WebSocket authentication beyond optional device allowlisting

## Phase 2

- Add persistent session and device repository
- Add proper WebSocket authentication and device provisioning
- Add distributed connection coordination if scaling beyond one worker
- Add real STT and TTS provider adapters
- Add external telemetry export such as OpenTelemetry, Loki, or ELK
- Add rollout-safe provider retry and circuit breaker policies
