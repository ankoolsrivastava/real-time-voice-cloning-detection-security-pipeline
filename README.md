# Real-Time Voice Anti-Spoofing & Impersonation Detection

A personal ML + backend engineering project for detecting **bonafide vs. spoofed speech** from incoming audio and converting per-window evidence into a **0–100 impersonation risk score**.

The system is focused on **Hindi and Marathi speech** and combines a custom PyTorch model with prosody evidence, audio-quality confidence, temporal risk tracking, and backend security decisions.

> **Project type:** Personal ML + Backend Engineering Project  
> **Status:** Working integrated prototype

---

## Overview

The system is designed for near-live speech analysis. Incoming audio is split into fixed analysis windows, converted to Log-Mel features, passed through a custom CNN + BiGRU + temporal-attention model, and then combined with secondary evidence before the backend applies a security policy.

```text
Audio Stream → Chunk Buffer → 10s Window → Log-Mel Features
      ↓
Custom CNN → BiGRU → Temporal Attention → Spoof Probability
      ↓
Prosody + Audio Quality → Risk Calculation → Temporal Tracking
      ↓
0–100 Risk Score → Security Policy → REST / WebSocket Response
```

### Main features
- Bonafide vs. spoof speech classification
- Hindi + Marathi speech
- Synthetic/TTS-style spoof detection
- Robustness evaluation with transformed and reverberant audio
- Fixed 10-second inference windows
- Custom CNN + BiGRU + temporal attention model
- Secondary prosody evidence
- Audio-quality confidence estimation
- Temporal risk tracking across windows
- 0–100 impersonation risk score
- FastAPI backend with REST and WebSocket interfaces
- Session management, telemetry, and automated tests

---

# Machine Learning

## Custom Model

The primary detector is a **custom PyTorch model implemented in this project**. It uses a Log-Mel spectrogram as input and learns both local time-frequency patterns and longer temporal patterns in speech.

### Architecture

```text
64-bin Log-Mel Spectrogram
          ↓
CNN: 1 → 32 → 64 → 128 → 128
          ↓
Adaptive Frequency Pooling
          ↓
2-Layer Bidirectional GRU
Hidden Size: 128
          ↓
Temporal Attention
          ↓
Classifier: 256 → 128 → 2
          ↓
BONAFIDE / SPOOF
```

### Model parameters

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Task | Binary classification |
| Classes | BONAFIDE / SPOOF |
| CNN channels | 1 → 32 → 64 → 128 → 128 |
| CNN kernel | 3 × 3 |
| CNN activation | GELU |
| Frequency pooling | Adaptive Average Pooling |
| Recurrent layer | 2-layer Bidirectional GRU |
| GRU hidden size | 128 |
| BiGRU output | 256 features |
| Attention | Learned temporal attention |
| Classifier | 256 → 128 → 2 |
| Total parameters | **1,143,331** |

The CNN extracts local time-frequency features, the BiGRU models temporal dependencies, and the attention layer learns which parts of the analysis window contribute more strongly to the final prediction.

## Frozen V2 checkpoint

The trained V2 checkpoint is the learned neural-network state produced from the locked V1 dataset pipeline.

```text
Checkpoint: best_model.pt
Model: voiceguard_v2_epoch8
Parameters: 1,143,331
Frozen spoof threshold: 0.25
SHA-256:
ad872bac0f754e554ede13507d86b2ed6396e3e4cbd7973935559c5c292d934b
```

The checkpoint does **not** contain the 17,511 source audio records. Raw audio remains intentionally outside GitHub because of its size and dataset-distribution considerations. With the checkpoint and supporting runtime artifacts, the frozen V2 inference/backend pipeline can run without downloading the full audio collection.

## Audio preprocessing

| Parameter | Value |
|---|---:|
| Sample rate | **16,000 Hz** |
| Channels | **Mono** |
| Data type | **Float32** |
| Normalization | Peak normalization |
| Window duration | **10 seconds** |
| Samples per window | **160,000** |
| FFT size | **400** |
| Hop length | **160** |
| Window length | **400** |
| Mel bins | **64** |
| Frequency range | **20–8,000 Hz** |
| Spectrogram | Log-Mel |

## Training configuration

| Parameter | Value |
|---|---:|
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Maximum epochs | 30 |
| Early stopping patience | 7 |
| Loss | Cross-Entropy |
| Random seed | 42 |

The training pipeline validates the manifest before training and requires both bonafide and spoof classes. It does not create dummy labels or placeholder training samples.

---

# Dataset

The project uses a locked V1 dataset covering bonafide and spoofed speech for Hindi and Marathi.

| Dataset component | Samples |
|---|---:|
| Bonafide speech (4,568 clean + 4,943 robustness) | 9,511 |
| Clean synthetic spoof | 3,000 |
| Spoof robustness speech | 5,000 |
| **Total** | **17,511** |

### Dataset coverage
- **9,657 Hindi samples**
- **7,854 Marathi samples**
- **13,214 training samples**
- **3,138 validation samples**
- **1,159 protected test samples**
- 4,568 bonafide clips from 424 speakers
- 16 kHz mono WAV source data
- Speaker-aware separation checks
- Dataset quality-control validation

The protected test split is kept separate from development and integration work. Raw audio and generated speech collections are intentionally excluded from the public repository because they are large dataset artifacts. The trained model checkpoint and supporting non-audio runtime artifacts are separate from the raw audio and can be distributed where appropriate.

---

# Evaluation

The frozen V2 model was evaluated using the documented project evaluation setup.

| Evaluation | Metric | Result |
|---|---|---:|
| Clean test | Accuracy @ 0.50 | **97.76%** |
| Clean test | ROC-AUC | **99.69%** |
| Clean test | EER | **1.98%** |
| Common 10K | Accuracy @ 0.25 | **96.34%** |
| Common 10K | F1 @ 0.25 | **96.40%** |
| Dual-RIR | Accuracy @ 0.25 | **92.90%** |
| Dual-RIR | F1 @ 0.25 | **93.00%** |

### Frozen threshold

**Spoof threshold: 0.25**

The threshold is part of the frozen V2 inference configuration and is a project-level operating point rather than a universal threshold for every deployment environment.

---

# Evidence and Risk

The primary acoustic detector is supplemented with secondary evidence.

```text
Primary V2 detector  → 83%
Prosody evidence     → 17%
Audio quality        → confidence / reliability
```

Audio quality is **not treated as spoof evidence**. Poor-quality audio reduces confidence in the available evidence instead of automatically being classified as spoof.

Speaker consistency was evaluated separately but is **not active in the frozen V2 pipeline** because its earlier validation did not justify using it as production evidence without a trusted speaker reference.

## Risk score

The evidence is converted into a normalized **0–100 impersonation risk score**.

| Score | Level |
|---:|---|
| 0–29 | LOW |
| 30–69 | MEDIUM |
| 70–89 | HIGH |
| 90–100 | CRITICAL |

| Risk condition | Example action |
|---|---|
| LOW | Allow |
| MEDIUM | Verify |
| HIGH | MFA / callback / escalation |
| CRITICAL | Block / escalation |
| Low-confidence evidence | Secondary verification |

The ML evidence layer and application security policy are kept separate.

---

# Backend

The backend is built with **FastAPI** and supports both REST and WebSocket communication.

### Streaming flow

```text
Audio Chunk → Input Validation → Chunk Buffer → Window Manager
      ↓
10-second Window → ML Runtime → Detection Result
      ↓
Temporal Risk Tracking → Security Policy → API / WebSocket
```

### Backend responsibilities
- Audio chunk validation
- Sequence numbers and timestamps
- Chunk buffering and fixed-window construction
- Session lifecycle management
- Persistent ML runtime
- ML connector layer
- Evidence/result serialization
- Temporal risk accumulation
- REST endpoints
- WebSocket streaming
- Transport telemetry
- Security-policy decisions
- Error handling and tests

The ML runtime loads the frozen model and supporting components once per backend process and reuses them for inference.

---

# Performance

Measured on an **NVIDIA RTX 3050** using the documented benchmark configuration:

| Metric | Result |
|---|---:|
| Mean end-to-end latency | **~13.06 ms** |
| Model-only latency | **~3.46 ms** |
| Peak VRAM | **~61.89 MB** |

These measurements apply to the tested hardware and configuration and are not a universal production performance guarantee.

---

# API

The backend provides endpoints for:
- Health checks
- Model information
- ML/backend integration information
- Session creation and inspection
- Audio chunk submission
- Session reset/deletion
- WebSocket streaming inference

---

# Repository structure

```text
real-time-voice-cloning-detection-security-pipeline/
├── backend/
│   ├── app/
│   │   ├── api/              # REST + WebSocket
│   │   ├── audio/            # Chunking and windows
│   │   ├── core/             # Backend core
│   │   ├── ml/               # Model integration and evidence
│   │   ├── security/         # Security policy
│   │   ├── services/         # Detection services
│   │   ├── sessions/         # Session management
│   │   ├── temporal/         # Risk tracking
│   │   └── transport/        # Telemetry
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
├── scripts/                  # Training, evaluation and ML utilities
├── 00_PROJECT_DOCS/          # Dataset/project reference material
├── handoff/                  # Final V2 ML handoff + verifier
├── ML_BACKEND_HANDOFF.md     # ML/backend integration reference
└── README.md
```

---

# Tech stack

### Machine Learning / Audio
- Python
- PyTorch
- TorchAudio
- Librosa
- NumPy
- SciPy
- Scikit-learn
- Pandas
- SoundFile
- SoundDevice

### Backend
- FastAPI
- Uvicorn
- Pydantic
- WebSockets
- Pytest

### Frontend
- React
- TypeScript
- Vite

The Python environment and dependency versions used by the backend are documented in `backend/requirements.txt`.

---

# Reproducibility

The repository is designed around three practical levels of reproducibility:

### 1. Clone only

A GitHub clone provides the ML source code, training/evaluation scripts, backend, frontend, tests, ML evidence contract, and final V2 handoff documentation. You can inspect the complete engineering pipeline and run tests that do not require unavailable audio/model artifacts.

### 2. Clone + dataset

With a compatible Hindi/Marathi dataset and the documented preprocessing/training pipeline, the training scripts can be used to train a **new model**. This does not reproduce the frozen V2 checkpoint unless the exact locked training data, split, environment, and configuration are available.

### 3. Clone + frozen ML artifacts

With the frozen `best_model.pt`, supporting prosody/runtime artifacts, and the documented handoff layout, the integrated backend can run the **frozen V2 inference pipeline** without the full 17,511-record audio collection.

**The raw audio dataset is the intentionally excluded large artifact; the trained model and other non-audio runtime artifacts are separate from it.**

---

# Local setup

## Backend

The backend expects the frozen ML artifacts to exist outside the repository. By default it looks for a sibling `ml_handoff/` directory next to `backend/`. Set `VOICEGUARD_ML_HANDOFF_ROOT` when the artifacts live elsewhere. The expected layout and verifier are documented in `handoff/`.

```powershell
cd backend
..\ml_env\Scripts\python.exe -m pip install -r requirements.txt
..\ml_env\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000/api`. Set `VITE_API_BASE_URL` if the backend is hosted elsewhere.

---

# Testing

Backend tests cover API behavior, audio buffering/window construction, ML connectors, and core integration.

```powershell
cd backend
..\ml_env\Scripts\python.exe -m pytest
```

The raw audio datasets are not included. Tests that exercise the frozen model require the corresponding model/runtime artifacts. The final V2 handoff package documents the required artifact layout and verifies the frozen checkpoint hash.

---

# Engineering notes

The project separates the main responsibilities into independent layers:

```text
Audio / Transport
       ↓
Streaming / Sessions
       ↓
ML Runtime
       ↓
Evidence
       ↓
Temporal Risk
       ↓
Security Policy
```

`ML_BACKEND_HANDOFF.md` contains the detailed ML ↔ backend integration contract and can be used when connecting the inference pipeline to another Python backend.

---

# Limitations

This is a **personal ML engineering and research-oriented prototype**, not a certified voice-authentication system.

- Spoof detection does not establish speaker identity.
- Performance can vary across microphones, codecs, channels, environments, and unseen synthesis methods.
- Prosody is a secondary signal.
- Audio quality affects confidence rather than directly determining spoof status.
- Broader external evaluation and calibration would be required for production deployment.
- Reported metrics and latency measurements correspond to the documented project evaluation and hardware configurations.

---

# ML handoff

The repository keeps large/private ML artifacts outside GitHub while retaining a reproducible integration boundary. The final V2 handoff package contains:

- `VOICEGUARD_FINAL_ML_HANDOFF_V2_README.md` — artifact layout, integration steps, and frozen identifiers
- `VOICEGUARD_FINAL_ML_HANDOFF_V2_INDEX.json` — machine-readable artifact manifest
- `VERIFY_VOICEGUARD_FINAL_ML_HANDOFF_V2.ps1` — local prerequisite/hash verifier

The handoff preserves the frozen V2 checkpoint SHA-256, prosody scorer configuration, runtime scripts, and ML evidence contract requirements without committing model weights or audio datasets.

---

# Author

**Ankool Srivastava**

GitHub: [ankoolsrivastava](https://github.com/ankoolsrivastava)