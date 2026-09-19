# Repository Working Rules

## Project scope

This repository is the personal implementation of a real-time voice anti-spoofing and impersonation-detection pipeline for Hindi and Marathi speech.

Keep the frozen ML contract intact unless a model/contract change is explicitly intended.

## Frozen ML rules

- Frozen model: `voiceguard_v2_epoch8`
- Frozen checkpoint SHA-256: `ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b`
- Primary threshold: `0.25`
- Input: 16 kHz mono float32 PCM
- Analysis window: 10 seconds / 160,000 samples
- Prosody is secondary evidence.
- Audio quality affects confidence and is not spoof evidence.
- Speaker consistency is inactive/unavailable without a trusted reference.
- Do not use the protected TEST split for tuning or integration work.
- Do not modify, relabel, resplit, reconvert, or retrain the locked V1/V2 artifacts without an explicit project decision.

## Repository hygiene

Do not commit:

- audio or dataset files
- model/checkpoint files
- virtual environments
- caches
- local logs
- generated experiment outputs
- secrets or credentials
- machine-specific handoff directories

The external ML handoff is documented under `handoff/`.

## Backend boundary

The backend owns transport, chunk buffering, exact window construction, sessions, telemetry, REST/WebSocket transport, temporal accumulation, and security-policy actions.

The external ML handoff owns the frozen model, preprocessing/runtime scripts, prosody scorer, quality interface, evidence fusion, and risk engine.

Do not silently duplicate or alter the ML contract in the backend.

## Development environment

The original local development root is on the D: drive. Keep project code, model artifacts, caches, and temporary project files off C:.

The expected backend environment is `ml_env`. Do not assume older environments such as `ai_env`.

## Verification

Before changing frozen ML behavior, inspect:

- `voiceguard_ml_evidence_contract_v1.json`
- `ML_BACKEND_HANDOFF.md`
- `handoff/VOICEGUARD_FINAL_ML_HANDOFF_V2_README.md`

After backend changes, run the backend tests with the configured local ML handoff available.
