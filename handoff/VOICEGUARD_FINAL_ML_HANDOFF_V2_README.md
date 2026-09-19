# Final V2 ML Handoff

This package defines the external ML artifact layout required by the FastAPI backend.

The repository intentionally excludes **audio datasets, generated outputs, and external runtime bundles**. The frozen checkpoint and final prosody scorer are intentionally published under `models/` for repository-level inspection and distribution. The external handoff directory remains the backend's deployment layout and is verified before startup.

## Frozen ML package

- Model: `voiceguard_v2_epoch8`
- Checkpoint: `models/best_model.pt`
- Checkpoint SHA-256: `ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b`
- Threshold: `0.25`
- Sample rate: 16 kHz
- Input: mono float32 PCM
- Analysis window: 10 seconds / 160,000 samples
- Languages: Hindi and Marathi
- Primary model: custom CNN + 2-layer BiGRU + temporal attention
- Prosody: secondary evidence, 83/17 frozen fusion configuration
- Speaker consistency: unavailable in the active pipeline
- Audio quality: confidence/reliability signal, not spoof evidence

## Required directory layout

Set `VOICEGUARD_ML_HANDOFF_ROOT` to the directory containing:

```text
ml_handoff/
├── models/
│   └── best_model.pt
├── prosody/
│   ├── prosody_scorer_v1.pkl
│   └── PROSODY_FUSION_CONFIG_V1.json
└── runtime/
    └── scripts/
        ├── audio_preprocessor.py
        ├── feature_extractor.py
        ├── model.py
        ├── voiceguard_prosody_fusion_v1.py
        ├── voiceguard_audio_quality_interface_v1.py
        └── voiceguard_dynamic_risk_engine_v2.py
```

The ML evidence contract is tracked in the repository as:

```text
voiceguard_ml_evidence_contract_v1.json
```

## How the backend uses it

The backend loads the handoff once during application startup:

```text
FastAPI
  ↓
backend/app/ml/runtime.py
  ↓
external ml_handoff/
  ├── frozen V2 checkpoint
  ├── prosody scorer
  └── runtime scripts
```

The backend remains responsible for microphone/call transport, buffering, fixed-window construction, sessions, telemetry, temporal accumulation, REST/WebSocket transport, and application security actions.

The ML handoff remains responsible for preprocessing, Log-Mel extraction, V2 inference, prosody evidence, audio-quality interpretation, evidence fusion, and per-window risk.

## Setup

From the repository root on Windows:

```powershell
$env:VOICEGUARD_ML_HANDOFF_ROOT = "D:\VoiceGaurd\ml_handoff"

cd backend
..\ml_env\Scripts\python.exe -m pip install -r requirements.txt
..\ml_env\Scripts\python.exe -m uvicorn app.main:app --reload
```

If the handoff directory is placed beside `backend/` as `<repo-root>\ml_handoff`, the environment variable is optional because that is the backend's default.

## Verify before starting

Run:

```powershell
.\handoff\VERIFY_VOICEGUARD_FINAL_ML_HANDOFF_V2.ps1 -HandoffRoot "D:\VoiceGaurd\ml_handoff"
```

The verifier checks the required files and the exact frozen V2 checkpoint SHA-256. It does not modify the checkpoint, datasets, or handoff artifacts.

## Integration contract

The authoritative ML-to-system schema is:

```text
voiceguard_ml_evidence_contract_v1
```

Important invariants:

- `spoof_probability >= 0.25` means the primary V2 decision is `SPOOF`.
- Class `0` is bonafide and class `1` is spoof.
- Prosody is secondary evidence, not a second final detector.
- Poor audio quality reduces evidence confidence; it does not become spoof evidence.
- Speaker consistency remains `UNAVAILABLE` without a trusted reference implementation.
- Risk is returned on a 0–100 scale with LOW / MEDIUM / HIGH / CRITICAL levels.
- The protected TEST split is not required by the runtime and is not included in this package.

## Change control

Do not replace the checkpoint, threshold, preprocessing configuration, prosody scorer, fusion weights, or risk semantics while keeping the same frozen identifiers.

Any such change requires a new artifact/version review and a corresponding contract update.
