# VoiceGuard V2 Evidence Integration V1

## Purpose

This is the integration smoke test for the current ML evidence stack.

It connects:

1. Frozen V2 inference
2. Audio quality interface
3. Prosody diagnostics
4. Dynamic risk engine

## Safety

Only `train` / `validation` samples are selected from the locked master manifest.
The protected `test` split is deliberately excluded.

No model or threshold tuning occurs.

No synthetic audio is generated.

## Important prosody limitation

The existing prosody experiment produced useful validation evidence, and a
validation fusion candidate was identified. However, its fitted prosody
scorer was not packaged as a standalone deployable calibrated artifact.

Therefore this integration test exposes real prosody diagnostic features but
does NOT manufacture a prosody probability/evidence score.

The risk engine consequently uses the frozen V2 detector + quality confidence
for this smoke test.

That is intentional and scientifically honest.

## Expected output

`VOICEGUARD V2 EVIDENCE INTEGRATION TEST PASS`

The script also writes:

`D:\VoiceGaurd\experiments\reverb_v2\evidence_integration\V2_EVIDENCE_INTEGRATION_TEST_V1.json`
