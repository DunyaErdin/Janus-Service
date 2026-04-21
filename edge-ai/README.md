# Janus Edge AI Service

Production-oriented Python edge orchestration service for a home assistant robot system. This service does not own low-level hardware. It accepts typed device events from the ESP runtime, maintains session context, interprets touch input semantically, coordinates placeholder speech/LLM/TTS providers, validates every outbound response plan, and returns only high-level semantic intents.

## Architectural Intent

- FastAPI app shell with a dedicated WebSocket transport for ESP connectivity
- Hexagonal architecture with explicit ports and adapters
- Strict Pydantic contracts for device protocol and AI response plans
- Thin transport layer, business flow centered in `ConversationOrchestrator`
- Safe fallback behavior whenever STT, LLM, TTS, or schema validation fails
- Provider boundaries that keep Gemini integration replaceable in the future

## Current Scope

Implemented:

- WebSocket endpoint for typed device messages
- Session lifecycle and in-memory session repository
- Semantic touch interpretation service
- Prompt builder with persona, enum constraints, and safety rules
- Mock LLM adapter that returns valid structured plans without external calls
- Placeholder Gemini, STT, and TTS adapters behind stable ports
- Response validation to prevent unsafe or low-level action emission
- JSON telemetry adapter for structured logs

Placeholders:

- Real STT transcription
- Real TTS synthesis or audio artifact dispatch
- Real Gemini transport and response parsing
- Authentication and device identity verification
- Persistent session storage
- External telemetry export

## Layout

```text
edge-ai/
  pyproject.toml
  README.md
  .env.example
  app/
    main.py
    config.py
    dependencies.py
    api/
      websocket_routes.py
    application/
      orchestrators/
        conversation_orchestrator.py
      services/
        fallback_response_service.py
        prompt_builder.py
        response_validator.py
        touch_interpreter.py
    domain/
      enums/
        action_type.py
        emotion.py
        face_expression.py
        voice_style.py
      models/
        ai_response_plan.py
        device_event.py
        session_context.py
        touch_context.py
      ports/
        llm_port.py
        provider_errors.py
        session_repository_port.py
        stt_port.py
        telemetry_port.py
        tts_port.py
    infrastructure/
      adapters/
        llm/
          gemini_llm_adapter.py
          mock_llm_adapter.py
        repositories/
          in_memory_session_repository.py
        stt/
          placeholder_stt_adapter.py
        telemetry/
          json_logger_telemetry_adapter.py
        tts/
          placeholder_tts_adapter.py
      transport/
        websocket/
          connection_manager.py
          protocol.py
    schemas/
      llm_response_schema.py
      websocket_messages.py
  tests/
    test_fallback_response_service.py
    test_response_validator.py
    test_touch_interpreter.py
```

## Protocol Notes

Incoming WebSocket messages are discriminated by `message_type` and include:

- `hello`
- `heartbeat`
- `touch_event`
- `audio_chunk`
- `session_start`
- `session_end`
- `status`

Outgoing WebSocket messages include:

- `ack`
- `ai_response_plan`
- `error`

The canonical response contract is `AIResponsePlan`, which always includes:

- `spoken_text`
- `emotion`
- `face_expression`
- `voice_style`
- `touch_interpretation`
- `actions`

## Provider Swappability

The orchestration layer depends only on ports:

- `LlmPort`
- `SttPort`
- `TtsPort`
- `DeviceSessionRepositoryPort`
- `TelemetryPort`

That means Gemini-specific logic can be completed or replaced later without touching the WebSocket handler, touch interpretation, response validation, or session orchestration flow.

## Notes For The Gemini Migration

`GeminiLlmAdapter` is included intentionally as a fail-fast scaffold. The rest of the service is already structured so a completed Gemini adapter can be dropped in behind `LlmPort` and selected via configuration. If you later move from Gemini to OpenAI, Anthropic, local vLLM, or another gateway, the swap should stay localized to the adapter and dependency wiring.

