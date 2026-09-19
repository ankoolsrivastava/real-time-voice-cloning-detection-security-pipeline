# VoiceGuard ML Evidence Contract V1

## Status

Integration-ready interface for the current Pre-SIH ML candidate.

## Active evidence

1. **V2 acoustic detector — PRIMARY**
   - `voiceguard_v2_epoch8`
   - frozen checkpoint
   - validation-selected threshold: `0.25`

2. **Prosody / behavioral evidence — SECONDARY**
   - validated as a candidate evidence branch
   - not a replacement for V2

3. **Audio quality — CONFIDENCE / RELIABILITY**
   - quality degradation lowers confidence
   - quality is never converted into spoof evidence

## Speaker consistency

The lightweight V1 experiment was not sufficiently discriminative:
- ROC-AUC: 0.5768
- EER: 45.76%

Therefore it is **not an active production evidence signal** in this freeze. The interface reserves the field for a future stronger speaker-verification implementation.

## Unavailable states

The contract explicitly supports:
- `UNAVAILABLE`
- `UNRELIABLE`
- `ERROR`

No-reference speaker consistency must be `UNAVAILABLE`.

## Streaming principle

A poor/corrupted window should reduce confidence rather than increase spoof evidence. The system should accumulate evidence from reliable windows instead of allowing one degraded window to dominate.

## Scope exclusions

- Indian English validation: pending
- Unseen voice-cloning evaluation: pending
