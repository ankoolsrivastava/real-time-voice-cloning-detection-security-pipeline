param(
    [Parameter(Mandatory = $false)]
    [string]$HandoffRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($HandoffRoot)) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $HandoffRoot = Join-Path $repoRoot "ml_handoff"
}

$requiredFiles = @(
    "models\best_model.pt",
    "prosody\prosody_scorer_v1.pkl",
    "prosody\PROSODY_FUSION_CONFIG_V1.json",
    "runtime\scripts\audio_preprocessor.py",
    "runtime\scripts\feature_extractor.py",
    "runtime\scripts\model.py",
    "runtime\scripts\voiceguard_prosody_fusion_v1.py",
    "runtime\scripts\voiceguard_audio_quality_interface_v1.py",
    "runtime\scripts\voiceguard_dynamic_risk_engine_v2.py"
)

$expectedSha256 = "ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b"

Write-Host "Final V2 ML handoff verification"
Write-Host "Handoff root: $HandoffRoot"
Write-Host ""

$missing = @()

foreach ($relativePath in $requiredFiles) {
    $fullPath = Join-Path $HandoffRoot $relativePath

    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
        $missing += $relativePath
        Write-Host "[MISSING] $relativePath"
    }
    else {
        Write-Host "[FOUND]   $relativePath"
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Error ("Missing required ML handoff files: " + ($missing -join ", "))
}

$modelPath = Join-Path $HandoffRoot "models\best_model.pt"
$actualSha256 = (Get-FileHash -LiteralPath $modelPath -Algorithm SHA256).Hash.ToLowerInvariant()

Write-Host ""
Write-Host "Checkpoint SHA-256:"
Write-Host "Expected: $expectedSha256"
Write-Host "Actual:   $actualSha256"

if ($actualSha256 -ne $expectedSha256) {
    Write-Error "Frozen V2 checkpoint SHA-256 does not match the recorded artifact."
}

Write-Host ""
Write-Host "[PASS] Final V2 ML handoff is complete and the frozen checkpoint hash matches."
