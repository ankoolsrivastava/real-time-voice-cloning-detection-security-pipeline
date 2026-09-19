$ErrorActionPreference = "Stop"

$root = "D:\VoiceGaurd"
$handoff = Join-Path $root "handoff"

$checkpoint = Join-Path $root "experiments\reverb_v2\models\best_model.pt"
$integration = Join-Path $root "experiments\reverb_v2\evidence_integration\V2_FULL_EVIDENCE_INTEGRATION_TEST_V2.json"
$index = Join-Path $handoff "VOICEGUARD_FINAL_ML_HANDOFF_V2_INDEX.json"

$expectedHash = "AD872BAC0F754E554EDE13507D86B2ED6396E3E4CBD7973935559C5C292D934B"

Write-Host "=== VoiceGuard FINAL ML HANDOFF V2 VERIFIER ===" -ForegroundColor Cyan

# 1. Check checkpoint
if (!(Test-Path $checkpoint)) {
    throw "FAIL: Frozen checkpoint missing."
}

$actualHash = (Get-FileHash $checkpoint -Algorithm SHA256).Hash

if ($actualHash -ne $expectedHash) {
    throw "FAIL: Frozen checkpoint SHA256 mismatch."
}

Write-Host "PASS: Frozen checkpoint SHA256 verified." -ForegroundColor Green

# 2. Check integration evidence
if (!(Test-Path $integration)) {
    throw "FAIL: V2 integration evidence missing."
}

$report = Get-Content $integration -Raw | ConvertFrom-Json

if ($report.status -ne "PASS") {
    throw "FAIL: V2 integration status is not PASS."
}

if ($report.protected_test_used -ne $false) {
    throw "FAIL: Protected test was used."
}

$requiredComponents = @(
    "frozen_v2",
    "deployable_prosody_scorer",
    "audio_quality",
    "dynamic_risk_engine_v2"
)

foreach ($component in $requiredComponents) {
    if ($report.active_components -notcontains $component) {
        throw "FAIL: Required component missing: $component"
    }
}

Write-Host "PASS: V2 integration evidence verified." -ForegroundColor Green

# 3. Check final index
if (!(Test-Path $index)) {
    throw "FAIL: Final handoff index missing."
}

$handoffIndex = Get-Content $index -Raw | ConvertFrom-Json

if ($handoffIndex.status -ne "FROZEN_FOR_BACKEND_HANDOFF") {
    throw "FAIL: Handoff status mismatch."
}

if ($handoffIndex.frozen_model.model_version -ne "voiceguard_v2_epoch8") {
    throw "FAIL: Model version mismatch."
}

if ([double]$handoffIndex.frozen_model.threshold -ne 0.25) {
    throw "FAIL: Threshold mismatch."
}

if ($handoffIndex.frozen_model.sha256.ToUpper() -ne $expectedHash) {
    throw "FAIL: Handoff hash mismatch."
}

Write-Host "PASS: Final handoff index verified." -ForegroundColor Green

Write-Host "`n=== FINAL RESULT: PASS ===" -ForegroundColor Green
Write-Host "VoiceGuard V2 ML handoff is internally consistent." -ForegroundColor Green
