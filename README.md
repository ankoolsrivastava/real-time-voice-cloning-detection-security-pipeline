# VaaniX

**Open-source Voice Integrity & Anti-Impersonation Platform**

VaaniX is a modular voice-security layer for detecting synthetic, cloned and spoofed speech in live or near-live voice interactions.

## Project status

The repository is being built around the validated VoiceGuard/VaaniX prototype: streaming ingestion, acoustic/spectral spoof detection, reliability telemetry, temporal risk accumulation, 0–100 operational risk scoring, configurable security policy, REST/WebSocket interfaces, sessions and automated backend tests.

The current frozen detector is VoiceGuard V2 epoch 8. Private/protected datasets, credentials and local model artifacts are not included.

## Vision

VaaniX is intended to grow into a complete open-source voice-security platform for consumer applications and enterprise deployments, with Indic-language robustness, real-world channel evaluation, pluggable detectors, calibrated risk, MFA/callback/escalation integrations, SDKs, deployment tooling, privacy controls and edge inference.

## Principles

- Be explicit about what is implemented versus proposed.
- Never commit private/protected audio or secrets.
- Make experiments and benchmarks reproducible.
- Keep the security layer modular and detector-agnostic.
- Treat model output as evidence, not an identity oracle.

See `ROADMAP.md`, `CONTRIBUTING.md`, `SECURITY.md` and `docs/` for project guidance.
