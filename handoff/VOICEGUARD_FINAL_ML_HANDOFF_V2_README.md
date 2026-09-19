# VoiceGuard Final ML Handoff V2

## Status

**FROZEN FOR BACKEND HANDOFF**

This package describes the actual current VoiceGuard ML implementation.

It does not represent planned, experimental, or legacy functionality.

## Frozen Model

- Model: `voiceguard_v2_epoch8`
- Checkpoint: `experiments\reverb_v2\models\best_model.pt`
- SHA256: `ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b`
- Threshold: `0.25`
- bonafide = `0`
- spoof = `1`

The frozen checkpoint must not be retrained, replaced, or recalibrated as part of backend integration.

## Active ML Pipeline

The active evidence path is:

Audio window
→ exact 16-kHz preprocessing
→ Log-Mel features
→ frozen V2 detector
→ spoof probability
→ deployable prosody scorer
→ audio quality/confidence
→ Dynamic Risk Engine V2
→ risk score / risk level / status

## Runtime Source

Required runtime files:

- `voiceguard_audio_quality_interface_v1.py`
- `voiceguard_prosody_fusion_v1.py`
- `voiceguard_dynamic_risk_engine_v2.py`
- `scripts\config.py`
- `scripts\model.py`
- `scripts\audio_preprocessor.py`
- `scripts\feature_extractor.py`
- `scripts\inference.py`

## Required Runtime Artifacts

- `experiments\reverb_v2\models\best_model.pt`
- `experiments\reverb_v2\prosody_fusion_v1\prosody_scorer_v1.pkl`
- `experiments\reverb_v2\prosody_fusion_v1\prosody_scorer_v1.json`
- `experiments\reverb_v2\prosody_fusion_v1\PROSODY_FUSION_CONFIG_V1.json`

## Risk Engine Interface

### Input

`RiskInput`:

- `spoof_probability`
- `quality_confidence_multiplier`
- `prosody_probability`
- `prosody_reliability`
- `evidence_confidence`

### Output

`RiskOutput`:

- `risk_score`
- `risk_level`
- `adjusted_spoof_probability`
- `primary_contribution`
- `prosody_contribution`
- `evidence_confidence`
- `status`
- `reasons`

Poor audio quality reduces confidence. It is not treated as spoof evidence.

## Verification

Verification runner:

`voiceguard_v2_full_evidence_integration_test_v2.py`

Verified result:

- status: `PASS`
- protected test used: `false`
- frozen V2: active
- deployable prosody scorer: active
- audio quality: active
- Dynamic Risk Engine V2: active

## Temporal Processing

Temporal risk accumulation is implemented as a supporting capability.

Live microphone/call transport and complete streaming integration remain backend/system responsibilities.

The current ML evidence does not claim complete end-to-end live streaming integration.

## Backend Responsibilities

Backend owns:

- microphone/call capture
- streaming transport
- buffering/windowing
- session management
- timestamps/window ordering
- network telemetry
- REST/WebSocket API
- frontend integration
- security policy/actions

## Backend Must Not Recreate

Do not independently recreate:

- preprocessing
- Log-Mel feature extraction
- prosody scoring
- audio-quality confidence
- evidence fusion
- dynamic risk calculation
- the frozen `0.25` ML threshold

The backend should consume the ML result.

## Legacy / Excluded

Do not use:

- `voiceguard_dynamic_risk_engine_v1.py`
- `test_dynamic_risk_engine_v1.py`
- `VOICEGUARD_DYNAMIC_RISK_ENGINE_V1_README.md`
- `voiceguard_v2_evidence_integration_test_v1.py`
- `VOICEGUARD_V2_EVIDENCE_INTEGRATION_V1_README.md`
- `V2_EVIDENCE_INTEGRATION_TEST_V1.json`
- Python caches / `.pyc` files

## Benchmark Note

The recorded production inference benchmark measures the real-file preprocessing + feature extraction + model pipeline on the tested hardware.

It is not an end-to-end microphone/network/streaming latency SLA.

## Source of Truth

This handoff is a snapshot of the actual frozen ML candidate.

Do not modify the model, threshold, or frozen dataset during backend integration unless the ML freeze is explicitly reopened.
