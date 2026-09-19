# VoiceGuard ML Handoff V1

## Decision

The core ML layer is ready to hand off to the backend/frontend team.

This is a handoff of the **frozen detector and validated integration contracts**.
It is not a claim that the entire end-to-end product is finished.

## Frozen ML path

Audio window
-> exact preprocessing
-> 64-bin Log-Mel
-> CNN
-> frequency-adaptive average pooling
-> 2-layer BiGRU
-> temporal attention
-> bonafide/spoof classifier
-> spoof probability
-> quality-aware evidence layer
-> Dynamic Risk Engine
-> 0-100 operational risk score

## What is active now

1. Frozen V2 epoch-8 detector.
2. Exact preprocessing and feature extraction.
3. Audio-quality interface.
4. Dynamic Risk Engine V1.
5. Temporal evidence accumulation mechanism.
6. Backend-facing evidence/output contract.

## What is not active

- Prosody classifier fusion is not claimed as deployable.
- Speaker consistency is rejected for the current deployment.
- Indian English has not yet been validated.
- Unseen/true cloned-voice generalization has not yet been validated.

## Backend/frontend responsibilities

The backend/frontend team can now implement:

- live microphone/call capture
- chunking/circular buffering
- packet-loss/jitter telemetry
- streaming inference API
- risk/action policy
- alerts/MFA/callback/escalation
- privacy/retention controls
- live UI

The ML team does not need to retrain V2 for these system tasks.

## Important integration rule

Poor audio/network quality must lower evidence confidence; it must never itself
be interpreted as spoof evidence.

## Files already created earlier

- `voiceguard_ml_evidence_contract_v1.json`
- `voiceguard_audio_quality_interface_v1.py`
- `voiceguard_dynamic_risk_engine_v1.py`
- `voiceguard_v2_evidence_integration_test_v1.py`

## Frozen checkpoint

`D:\VoiceGaurd\experiments\reverb_v2\models\best_model.pt`

SHA256:

`ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b`

## Handoff rule

Do not modify the locked master dataset or frozen V2 checkpoint as part of
backend/frontend integration.
