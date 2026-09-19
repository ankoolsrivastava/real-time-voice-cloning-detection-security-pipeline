# VoiceGuard Dynamic Risk Engine V1

## Purpose

This component is the transparent decision/evidence layer above the frozen
VoiceGuard V2 epoch-8 detector.

It does not retrain or modify the detector.

## Frozen model reference

- Model: `voiceguard_v2_epoch8`
- Threshold: `0.25`
- Primary detector weight: `0.89`
- Prosody secondary weight: `0.11`

The 0.89/0.11 fusion weight is retained as the current candidate from the
validation-only prosody incremental experiment. It is an engineering
integration parameter, not a new unbiased test result.

## Evidence rules

1. Primary spoof probability is the main evidence.
2. Prosody is optional secondary evidence.
3. Audio quality/network degradation never becomes spoof evidence.
4. Degraded quality shrinks evidence toward neutral (0.5 probability) and
   marks the result `LOW_CONFIDENCE`.
5. Low-confidence windows can be skipped by `TemporalRiskAccumulator`.
6. Speaker consistency is intentionally not used.
7. The engine outputs a 0-100 risk score and LOW/MEDIUM/HIGH/CRITICAL level.

## Risk levels

- LOW: `< 30`
- MEDIUM: `30–69.9999`
- HIGH: `70–89.9999`
- CRITICAL: `>= 90`

These are engineering policy bands, not statistically calibrated probabilities.

## Input semantics

### spoof_probability
`0..1`, directly from the frozen V2 detector.

### quality_confidence_multiplier
`0..1`, from the audio-quality interface.

### prosody_evidence
Optional `-1..+1`.
- `-1`: evidence toward bonafide
- `0`: neutral
- `+1`: evidence toward spoof

### prosody_reliability
`0..1`; controls how strongly optional prosody evidence can affect risk.

### evidence_confidence
`0..1`; additional confidence supplied by the evidence layer.

## Near-live behavior

Use one `RiskInput` per reliable audio window. Add the resulting
`RiskOutput` to `TemporalRiskAccumulator`.

Low-confidence windows are not accumulated. Reliable windows are combined
with confidence-weighted averaging.

## Not a calibration claim

The 0–100 score is an operational risk score, not a calibrated probability
of impersonation. Thresholds and policy bands should be revalidated on a
representative protected evaluation set before production deployment.
