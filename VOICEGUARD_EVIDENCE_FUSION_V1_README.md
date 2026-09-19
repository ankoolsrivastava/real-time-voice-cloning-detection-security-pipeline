# VoiceGuard Evidence Fusion V1

This package closes two current ML integration gaps.

## Gap 1: Prosody fusion not deployed

`build_voiceguard_prosody_fusion_v1.py` creates a real deployable secondary
prosody scorer.

Protocol:
- fit on TRAIN only
- select V2/prosody fusion weight on VALIDATION only
- never read protected TEST during fitting or selection
- save a versioned scorer artifact and JSON configuration

The scorer uses:
- duration
- RMS mean/std
- RMS dynamic range
- pause total/mean
- voiced/activity ratio
- voiced transition rate

## Gap 2: Evidence/quality layer partial integration

`voiceguard_dynamic_risk_engine_v2.py` actively consumes:
- frozen V2 spoof probability
- deployable prosody probability
- quality confidence
- overall evidence confidence

Quality attenuates evidence toward neutral. It never creates spoof evidence.

## Important

The validation-selected fusion weight is an engineering candidate, not a new
unbiased test result. The protected test should only be evaluated after the
artifact and configuration are frozen.

## Live microphone clarification

Microphone/call capture is NOT part of the neural ML model.

The system layer/backend/streaming component captures audio and produces
analysis windows plus network metadata. The ML inference layer consumes those
windows.

So the eventual boundary is:

`Microphone/call -> streaming/backend -> ML inference -> evidence/risk -> backend action`

The ML side owns preprocessing, feature extraction, model inference, prosody,
quality interpretation, evidence fusion and risk computation. The backend owns
capture/transport, session management, API orchestration and security actions.
