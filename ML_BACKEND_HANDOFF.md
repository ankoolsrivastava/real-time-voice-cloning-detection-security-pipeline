# V2 ML → Backend Integration Handoff

**Status:** Integration Ready
**ML model:** `voiceguard_v2_epoch8`
**Contract:** `voiceguard_ml_evidence_contract_v1`
**Threshold:** `0.25`
**Sample rate:** `16,000 Hz`
**Maximum analysis window:** `10 seconds`

---

## 1. Purpose

This document defines the stable interface between the VoiceGuard ML layer and the backend/system layer.

The backend supplies validated audio windows and stream context to the production ML predictor.

The ML layer returns structured acoustic evidence, audio-quality information, evidence confidence, and dynamic risk.

This handoff does not include microphone capture, call transport, API implementation, frontend implementation, or security-action implementation.

The authoritative contract is:

`voiceguard_ml_evidence_contract_v1`

The production ML entry point is:

`voiceguard_ml_predictor_v1.py`

---

## 2. Current ML Pipeline

The production ML path is:

```text
PCM float32 mono waveform
        ↓
VoiceGuard audio preprocessing
        ↓
Log-Mel feature extraction
        ↓
Frozen VoiceGuard V2
        ↓
Spoof probability
        ↓
Prosody evidence
        ↓
Audio quality assessment
        ↓
Evidence fusion / dynamic risk engine
        ↓
Structured ML result
```

VoiceGuard V2 is the primary acoustic detector.

Prosody is secondary evidence.

Audio quality controls evidence confidence and is not itself spoof evidence.

Speaker consistency is currently unavailable.

---

## 3. Frozen Production Artifacts

### 3.1 Primary model

```text
Model version:
voiceguard_v2_epoch8
```

Checkpoint:

```text
experiments/reverb_v2/models/best_model.pt
```

Checkpoint SHA-256:

```text
ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b
```

### 3.2 Frozen threshold

```text
0.25
```

Decision rule:

```text
spoof_probability >= 0.25 → SPOOF
spoof_probability <  0.25 → BONAFIDE
```

### 3.3 Class mapping

```text
0 → bonafide
1 → spoof
```

### 3.4 Prosody scorer

```text
experiments/reverb_v2/prosody_fusion_v1/prosody_scorer_v1.pkl
```

The active prosody configuration is loaded from its accompanying metadata.

The backend must not independently reproduce or modify the ML fusion calculation.

---

## 4. Public ML Interface

The repository contains `voiceguard_ml_predictor_v1.py` as the standalone ML-facing reference interface. The integrated FastAPI backend uses `backend/app/ml/runtime.py`, which loads the same frozen model, prosody scorer, quality interface, and risk engine from the external `ml_handoff/` package. Both paths are required to preserve the same ML contract.

The public prediction method is:

```python
predictor.predict(
    waveform,
    sample_rate=16000,
    window_id=window_id,
    timestamp_start_ms=timestamp_start_ms,
    timestamp_end_ms=timestamp_end_ms,
    sequence_id=sequence_id,
    quality_context=quality_context,
    speaker_reference=None,
)
```

The backend runtime should not duplicate model weights or training artifacts. It loads the frozen runtime modules from the external handoff package and keeps transport/session orchestration in the backend.

---

## 5. Input Contract

### 5.1 Audio

| Field            | Requirement        |
| ---------------- | ------------------ |
| `waveform`       | Mono PCM waveform  |
| dtype            | float32-compatible |
| channels         | 1                  |
| sample rate      | exactly 16,000 Hz  |
| maximum duration | 10 seconds         |
| maximum samples  | 160,000            |

The predictor validates the audio before inference.

The production preprocessor performs the established VoiceGuard preprocessing path, including mono/16-kHz processing, peak normalization, and fixed 10-second pad/truncate behavior.

### 5.2 Stream context

The following fields are required:

```text
window_id
timestamp_start_ms
timestamp_end_ms
sequence_id
```

These identify the analysis window and its position in the stream.

### 5.3 Quality context

Quality context is optional.

Supported fields are:

```python
{
    "packet_loss_ratio": float,
    "jitter_ms": float,
    "codec_degradation_score": float
}
```

Each field is optional.

When supplied, these values are passed into the audio-quality interface.

They influence evidence confidence/risk handling.

They do not become independent spoof evidence.

Unsupported quality-context fields are rejected by the production predictor.

---

## 6. Speaker Reference

The public interface contains:

```python
speaker_reference=None
```

This field is reserved for future trusted speaker-reference functionality.

### Current behavior

Speaker consistency is not implemented as an active production evidence source.

The current production result reports:

```json
{
  "available": false,
  "similarity": null,
  "evidence": null,
  "status": "UNAVAILABLE"
}
```

The backend must not interpret `UNAVAILABLE` as either a match or mismatch.

The backend must not create its own speaker-consistency score and insert it into the ML risk result.

---

## 7. Output Contract

A successful prediction returns:

```text
model_version
window_id
sequence_id
timestamp_start_ms
timestamp_end_ms
spoof_probability
threshold
primary_decision
prosody
speaker_consistency
audio_quality
evidence_confidence
risk
notes
```

### 7.1 Primary result

```json
{
  "spoof_probability": 0.0,
  "threshold": 0.25,
  "primary_decision": "BONAFIDE"
}
```

`spoof_probability` is the frozen V2 probability for class 1 (`spoof`).

### 7.2 Context fields

The output preserves:

```text
window_id
sequence_id
timestamp_start_ms
timestamp_end_ms
```

The backend should use these fields to associate the result with the corresponding stream window.

---

## 8. Prosody Result

Structure:

```json
{
  "available": true,
  "evidence": 0.0,
  "reliability": 1.0,
  "status": "AVAILABLE"
}
```

Possible status values:

```text
AVAILABLE
UNRELIABLE
ERROR
```

Prosody is secondary evidence.

It must not be treated by the backend as a second independent final detector.

---

## 9. Speaker Consistency Result

Structure when unavailable:

```json
{
  "available": false,
  "similarity": null,
  "evidence": null,
  "status": "UNAVAILABLE"
}
```

Possible status values:

```text
MATCH
MISMATCH
UNAVAILABLE
UNRELIABLE
ERROR
```

For the current production ML implementation, the expected state is:

```text
available = false
status = UNAVAILABLE
```

unless a future explicitly approved implementation changes this behavior.

---

## 10. Audio Quality

Structure:

```json
{
  "quality_score": 0.0,
  "confidence_multiplier": 0.0,
  "status": "GOOD"
}
```

The production result may additionally expose diagnostic fields such as:

```text
rms_db
peak
clipping_ratio
duration_sec
active_ratio
```

Possible quality states:

```text
GOOD
DEGRADED
POOR
INVALID
```

### Important rule

Poor audio quality does not mean spoof.

Quality degradation reduces confidence in the available evidence.

The backend must not convert:

```text
POOR quality
```

into:

```text
SPOOF
```

---

## 11. Evidence Confidence

```text
evidence_confidence ∈ [0,1]
```

This represents the confidence/reliability of the evidence under the current audio-quality and evidence conditions.

Quality degradation can attenuate evidence toward neutral.

The backend should use this field as returned by the ML layer and should not independently recalculate it.

---

## 12. Risk Output

The ML layer returns:

```json
{
  "risk_score": 0.0,
  "risk_level": "LOW",
  "adjusted_spoof_probability": 0.0,
  "primary_contribution": 0.0,
  "prosody_contribution": 0.0,
  "status": "USABLE",
  "reasons": []
}
```

### 12.1 Risk score

Range:

```text
0–100
```

### 12.2 Risk levels

```text
0–29.999    → LOW
30–69.999   → MEDIUM
70–89.999   → HIGH
90–100      → CRITICAL
```

### 12.3 Risk calculation ownership

The risk engine is part of the ML/evidence layer.

The backend should consume the returned risk result rather than recreate the risk calculation.

The backend should not modify the risk score using its own spoof/prosody weights.

---

## 13. Complete Output Shape

Conceptually, a successful result has this structure:

```json
{
  "model_version": "voiceguard_v2_epoch8",
  "window_id": "example_window",
  "sequence_id": 1,
  "timestamp_start_ms": 0,
  "timestamp_end_ms": 10000,

  "spoof_probability": 0.0,
  "threshold": 0.25,
  "primary_decision": "BONAFIDE",

  "prosody": {
    "available": true,
    "evidence": 0.0,
    "reliability": 1.0,
    "status": "AVAILABLE"
  },

  "speaker_consistency": {
    "available": false,
    "similarity": null,
    "evidence": null,
    "status": "UNAVAILABLE"
  },

  "audio_quality": {
    "quality_score": 1.0,
    "confidence_multiplier": 1.0,
    "status": "GOOD"
  },

  "evidence_confidence": 1.0,

  "risk": {
    "risk_score": 0.0,
    "risk_level": "LOW",
    "adjusted_spoof_probability": 0.0,
    "primary_contribution": 0.0,
    "prosody_contribution": 0.0,
    "status": "USABLE",
    "reasons": []
  },

  "notes": []
}
```

The numerical values above are structural examples only and are not benchmark results or model-performance claims.

---

## 14. Example Backend Call

```python
from voiceguard_ml_predictor_v1 import VoiceGuardMLPredictor

predictor = VoiceGuardMLPredictor()

result = predictor.predict(
    waveform=audio_window,
    sample_rate=16000,
    window_id="call_00017_w0042",
    timestamp_start_ms=41000,
    timestamp_end_ms=51000,
    sequence_id=42,
    quality_context={
        "packet_loss_ratio": 0.02,
        "jitter_ms": 10.0,
        "codec_degradation_score": 0.10,
    },
)

print(result["primary_decision"])
print(result["risk"]["risk_level"])
```

The returned dictionary is the ML result for that window.

---

## 15. Backend Responsibilities

The backend/system layer owns:

* microphone capture
* call/audio transport
* buffering
* creation of analysis windows
* timestamps
* sequence numbering
* session management
* API endpoints
* authentication
* authorization
* rate limiting
* application security
* temporal accumulation/orchestration
* frontend communication
* alerts
* security/application actions
* lifecycle management of the ML predictor

The backend should provide the ML layer with correctly identified audio windows.

---

## 16. ML Responsibilities

The ML layer owns:

* audio preprocessing
* Log-Mel feature extraction
* frozen V2 inference
* spoof probability
* primary threshold decision
* prosody extraction/scoring
* audio-quality interpretation
* evidence fusion
* dynamic risk calculation
* structured ML output

The backend should not duplicate these calculations.

---

## 17. Streaming Boundary

The ML predictor operates on individual audio windows.

It does not perform:

```text
microphone capture
call capture
network streaming
session management
```

The backend/system layer is responsible for supplying successive windows.

Conceptually:

```text
Microphone / Call
       ↓
Backend capture
       ↓
Window 1 ──→ ML predictor
Window 2 ──→ ML predictor
Window 3 ──→ ML predictor
       ↓
Backend temporal/session layer
       ↓
Application decision / UI / security action
```

Temporal accumulation is therefore a system/backend orchestration responsibility around the per-window ML result.

---

## 18. Error Handling

The predictor validates:

* sample rate
* waveform shape
* non-empty input
* finite values
* maximum window length
* required stream context
* supported quality-context fields

Invalid inputs can raise an exception.

The backend should catch ML-layer exceptions at its integration boundary and convert them into the backend's normal error response.

The backend must not silently fabricate a spoof probability or risk score when ML inference fails.

For a failed ML inference, the system should represent the result as an inference failure rather than inventing:

```text
SPOOF
```

or:

```text
BONAFIDE
```

---

## 19. Concurrency

The predictor loads the frozen model and supporting artifacts during initialization.

Recommended integration pattern:

```text
Application startup
       ↓
Initialize ML predictor
       ↓
Reuse predictor for inference requests
```

Do not repeatedly reload the checkpoint for every audio window.

The backend should control request/session concurrency according to the available hardware and expected workload.

The ML layer should remain an inference component; concurrency and session orchestration belong to the backend.

---

## 20. What Backend Must NOT Change

Without an explicit ML review, the backend must not change:

```text
V2 checkpoint
threshold = 0.25
class mapping
Log-Mel configuration
preprocessing behavior
prosody scorer
prosody fusion weights
risk-engine calculations
speaker-consistency semantics
ML output field meanings
```

The backend must also not:

* retrain the model
* alter the frozen dataset
* use the protected TEST split for tuning
* replace the ML threshold with an arbitrary application threshold
* treat prosody as an independent final detector
* treat poor quality as spoof evidence
* fabricate missing ML evidence
* implement its own speaker-verification score

---

## 21. When Backend Changes Must Come Back to ML

Backend development should trigger an ML-contract review if it requires any change to:

* audio sample rate
* audio channel assumptions
* maximum analysis-window size
* preprocessing
* ML input fields
* ML output fields
* threshold
* model checkpoint
* feature configuration
* prosody configuration
* evidence weights
* risk semantics
* speaker-consistency behavior

Such changes require explicit contract/version review before integration.

Do not silently change an existing contract while keeping the same version identifier.

---

## 22. Current Known Limitations / Pending Work

The following are intentionally not represented as completed:

### Streaming integration

The ML predictor is ready for backend invocation, but actual microphone/call streaming integration remains a system/backend task.

### Temporal stream hookup

The predictor returns per-window evidence.

System-level temporal accumulation across windows remains part of the backend/application integration.

### Speaker consistency

Speaker consistency remains:

```text
UNAVAILABLE
```

unless a future approved trusted-reference implementation is added.

### Current language scope

The current frozen project scope is Hindi and Marathi. Indian English is not part of the active target or evaluation scope.

### Unseen cloning/generalization

Evaluation against actual unseen cloned-voice generators remains a planned validation extension.

These pending items do not justify changing the frozen V2 model as part of this handoff.

---

## 23. Version / Change Control

Current production-candidate ML package:

```text
Model:
voiceguard_v2_epoch8

Contract:
voiceguard_ml_evidence_contract_v1
```

Frozen threshold:

```text
0.25
```

Any change to the ML artifacts or contract should be explicitly identified.

Examples:

```text
new checkpoint
→ model version review

new threshold
→ decision-contract review

new output field
→ backend contract review

changed risk logic
→ evidence/risk contract review

changed preprocessing
→ ML/backend compatibility review
```

Do not silently change an existing contract while keeping the same version identifier.

---

## 24. Integration Summary

The intended integration boundary is:

```text
                BACKEND
                   │
                   │ validated PCM + context
                   ▼
        ┌──────────────────────┐
        │ VoiceGuardMLPredictor│
        └──────────┬───────────┘
                   │
          ┌────────▼────────┐
          │ Frozen V2       │
          │ + Prosody       │
          │ + Quality       │
          │ + Risk          │
          └────────┬────────┘
                   │
                   │ structured result
                   ▼
                BACKEND
                   │
          temporal/session logic
                   │
          UI / alerts / actions
```

The backend integrates with the predictor.

The backend does not need to know the internal V2 implementation.

---

## 25. Final Status

```text
Frozen V2 model              READY
Production predictor         READY
Prosody evidence             READY
Audio quality interface      READY
Evidence fusion              READY
Dynamic risk engine          READY
ML evidence contract         READY
Backend handoff              READY

Microphone/call integration  BACKEND TASK
Temporal stream hookup       BACKEND TASK
Security actions             BACKEND TASK

Speaker consistency          UNAVAILABLE
Indian English validation    PENDING
Unseen cloning evaluation    PENDING
```

This document describes the current production ML handoff and should be treated together with:

```text
voiceguard_ml_evidence_contract_v1.json
```

as the integration reference.
