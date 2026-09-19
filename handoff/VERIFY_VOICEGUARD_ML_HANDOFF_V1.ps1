$ErrorActionPreference = "Stop"

$Root = "D:\VoiceGaurd"
$Checkpoint = Join-Path $Root "experiments\reverb_v2\models\best_model.pt"
$Master = Join-Path $Root "master_v1\metadata\master_dataset_v1.csv"
$Integration = Join-Path $Root "experiments\reverb_v2\evidence_integration\V2_EVIDENCE_INTEGRATION_TEST_V1.json"

$ExpectedSha = "ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b"

Write-Host "VOICEGUARD ML HANDOFF VERIFICATION V1"
Write-Host "==================================="

if (!(Test-Path $Checkpoint)) { throw "Frozen V2 checkpoint missing: $Checkpoint" }
if (!(Test-Path $Master)) { throw "Master manifest missing: $Master" }
if (!(Test-Path $Integration)) { throw "Integration report missing: $Integration" }

$ActualSha = (Get-FileHash $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()

Write-Host "Checkpoint SHA256:"
Write-Host $ActualSha

if ($ActualSha -ne $ExpectedSha) {
    throw "CHECKPOINT HASH MISMATCH. STOP."
}

$Rows = (Import-Csv $Master).Count
if ($Rows -ne 17511) {
    throw "MASTER MANIFEST ROW COUNT CHANGED: $Rows"
}

$IntegrationReport = Get-Content $Integration -Raw | ConvertFrom-Json
if ($IntegrationReport.status -ne "PASS") {
    throw "Integration report is not PASS."
}

Write-Host ""
Write-Host "Checkpoint hash       : PASS"
Write-Host "Master rows = 17511   : PASS"
Write-Host "Integration report    : PASS"
Write-Host ""
Write-Host "VOICEGUARD ML HANDOFF VERIFICATION PASS"
