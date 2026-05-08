# ADR 0001: Wake Detection Strategy

The MVP wake detector is server-side and STT-based. The ESP32-S3 sends bounded
PCM16 wake windows after a local activity prefilter; the edge service confirms
only when transcription normalizes to "hey janus" or close expected variants.

`EDGE_AI_WAKE_DETECTOR_PROVIDER=stt` is the production default. The
`dev_fake` provider is deterministic test tooling only and must not be treated
as production wake-word recognition.

Raw wake audio is not stored by default. Any debug dumps must be explicitly
enabled and treated as sensitive user audio.
