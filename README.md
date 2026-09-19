# Real-Time Voice Anti-Spoofing & Impersonation Detection

A real-time / near-live machine learning and backend system for detecting **bonafide vs. spoofed speech** and converting audio evidence into an operational **0–100 impersonation risk score**.

The system is designed for **Hindi and Marathi speech** and combines a custom PyTorch anti-spoofing model, prosodic evidence, audio-quality confidence, temporal risk accumulation, and security-policy actions.

> **Project type:** Personal ML + Backend Engineering Project  
> **Status:** Working integrated prototype  
> **Primary task:** Speech-level spoof / synthetic-voice detection

---

## What This Project Does

The system processes incoming speech in fixed streaming windows and evaluates whether the audio is consistent with **bonafide speech or spoofed speech**.

It is designed around the following pipeline:

```text
Microphone / Audio Stream
        ↓
Chunk Buffering
        ↓
Window Management
        ↓
16 kHz Mono Audio
        ↓
Log-Mel Spectrogram
        ↓
Custom CNN + BiGRU + Temporal Attention Model
        ↓
Primary Spoof Probability
        ↓
Prosody Evidence + Audio Quality
        ↓
Evidence / Risk Computation
        ↓
Temporal Risk Accumulation
        ↓
0–100 Impersonation Risk Score
        ↓
Security Policy
        ↓
REST / WebSocket Response
```

The backend is responsible for streaming, buffering, sessions, transport, telemetry, and security actions while the ML runtime keeps the model preprocessing and inference contract consistent.

---

## Core Capabilities

- Real-time / near-live audio processing
- Bonafide vs. spoof classification
- Hindi + Marathi speech support
- Synthetic / TTS-style spoof detection
- Robustness evaluation under transformed and reverberant audio conditions
- Fixed 10-second inference windows
- CNN + BiGRU + temporal attention architecture
- Prosody-based secondary evidence
- Audio-quality confidence estimation
- Temporal risk accumulation across multiple windows
- 0–100 impersonation risk scoring
- REST API
- WebSocket streaming
- Session management
- Structured telemetry
- Security-policy decisions
- ML ↔ backend integration contract
- Automated backend tests

---

# Machine Learning Pipeline

## Custom Model

The primary detector is a **custom PyTorch model implemented in this repository** rather than a pretrained end-to-end voice-classification model.

### Architecture

```text
64-bin Log-Mel Spectrogram
            ↓
        CNN Block
       1 → 32 channels
            ↓
        CNN Block
      32 → 64 channels
            ↓
        CNN Block
     64 → 128 channels
            ↓
        CNN Block
    128 → 128 channels
            ↓
 Adaptive Frequency Pooling
            ↓
      2-Layer BiGRU
       Hidden Size 128
            ↓
    Temporal Attention
            ↓
      Classification Head
            ↓
     BONAFIDE / SPOOF
```

### Model Details

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Task | Binary classification |
| Classes | BONAFIDE / SPOOF |
| CNN channels | 1 → 32 → 64 → 128 → 128 |
| Recurrent layer | 2-layer bidirectional GRU |
| GRU hidden size | 128 |
| BiGRU output size | 256 |
| Attention | Learned temporal attention |
| Classifier | 256 → 128 → 2 |
| Parameters | **1,143,331** |

The CNN learns local time-frequency patterns from the spectrogram, the BiGRU models temporal dependencies, and the attention layer learns which temporal regions contribute most strongly to the final decision.

---

## Audio Preprocessing Contract

The same preprocessing contract is used by the frozen inference runtime and backend integration:

| Parameter | Value |
|---|---:|
| Sample rate | **16,000 Hz** |
| Channels | **Mono** |
| Data type | **Float32** |
| Normalization | Peak normalization |
| Maximum window | **10 seconds** |
| Samples / window | **160,000** |
| FFT size | **400** |
| Hop length | **160** |
| Window length | **400** |
| Mel bins | **64** |
| Minimum frequency | **20 Hz** |
| Maximum frequency | **8,000 Hz** |
| Spectral representation | Log-Mel |

The backend constructs the exact inference window before passing the waveform to the ML runtime.

---

## Training Configuration

The model training pipeline is implemented in Python/PyTorch and reads from the locked master dataset rather than generating placeholder training data.

| Configuration | Value |
|---|---:|
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Maximum epochs | 30 |
| Early stopping patience | 7 |
| Random seed | 42 |
| Loss | Cross-Entropy |

The training code also validates the dataset before training and refuses to proceed when both required classes are not available.

---

# Dataset

The project uses a **locked internal V1 dataset** covering bonafide and spoofed speech for Hindi and Marathi.

### Dataset Composition

| Component | Samples |
|---|---:|
| Bonafide speech | 9,511 |
| Clean synthetic spoof | 3,000 |
| Real robustness speech | 4,943 |
| Spoof robustness speech | 5,000 |
| **Total** | **17,511** |

### Dataset Coverage

- **9,657 Hindi samples**
- **7,854 Marathi samples**
- **13,214 training samples**
- **3,138 validation samples**
- **1,159 protected test samples**
- 4,568 bonafide clips from 424 speakers
- 16 kHz mono WAV source data
- Speaker-aware separation checks
- Quality-control validation across the locked dataset

The protected test split is kept separate from development and integration work.

Raw audio, generated speech collections, and protected evaluation data are **not stored in the public repository**.

---

# Model Evaluation

The frozen V2 model was evaluated against the locked evaluation setup.

| Evaluation | Metric | Result |
|---|---|---:|
| Clean test | Accuracy @ 0.50 | **97.76%** |
| Clean test | ROC-AUC | **99.69%** |
| Clean test | EER | **1.98%** |
| Common 10K | Accuracy @ 0.25 | **96.34%** |
| Common 10K | F1 @ 0.25 | **96.40%** |
| Dual-RIR | Accuracy @ 0.25 | **92.90%** |
| Dual-RIR | F1 @ 0.25 | **93.00%** |

### Frozen Inference Threshold

**Spoof threshold: 0.25**

The threshold is part of the frozen V2 inference configuration. It is not presented as a universal industry threshold; operational behavior depends on the evaluation and deployment conditions.

---

## Inference Benchmark

Measured on an **NVIDIA RTX 3050** using the documented benchmark configuration:

| Metric | Measurement |
|---|---:|
| Mean end-to-end latency | **~13.06 ms** |
| Model-only latency | **~3.46 ms** |
| Peak VRAM | **~61.89 MB** |

These measurements describe the tested configuration and hardware rather than a universal production performance guarantee.

---

# Evidence & Risk Layer

The primary detector is combined with secondary speech evidence.

### Evidence Fusion

```text
Primary V2 Acoustic/Spectral Evidence  → 83%
Prosody Evidence                       → 17%
Audio Quality                          → Confidence / Reliability
```

Audio quality is intentionally treated as a **confidence signal**, not as independent spoof evidence.

Speaker-consistency scoring was evaluated separately but is **not active in the frozen V2 pipeline** because its earlier validation performance did not justify using it as production evidence without a trusted reference.

---

## Dynamic Risk Engine

The system converts the resulting evidence into a normalized **0–100 risk score**.

| Score | Risk Level |
|---:|---|
| 0–24 | LOW |
| 25–49 | MEDIUM |
| 50–74 | HIGH |
| 75–100 | CRITICAL |

The backend then applies a separate security policy to the risk result.

### Example Security Actions

| Risk condition | Backend action |
|---|---|
| Low risk | ALLOW |
| Medium risk | VERIFY |
| High risk | MFA / callback / escalation |
| Critical risk | BLOCK / escalation |
| Low-confidence evidence | Secondary verification |

The policy layer deliberately separates **ML evidence** from **application security decisions**.

---

# Real-Time Backend

The backend is implemented with FastAPI and provides both REST and WebSocket interfaces.

### Streaming Flow

```text
Audio Chunk
    ↓
Pydantic Validation
    ↓
Chunk Buffer
    ↓
Window Manager
    ↓
Exact 10-second Window
    ↓
ML Runtime
    ↓
Detection Result
    ↓
Temporal Accumulator
    ↓
Security Policy
    ↓
API / WebSocket Response
```

### Backend Responsibilities

- Audio chunk validation
- Sequence and timestamp handling
- Chunk buffering
- Exact-window construction
- Session lifecycle management
- Persistent ML runtime
- Model connector abstraction
- Evidence serialization
- Temporal risk accumulation
- Risk trend tracking
- Security-policy evaluation
- REST endpoints
- WebSocket streaming
- Telemetry
- Error handling
- Backend tests

The ML runtime loads the frozen model and supporting artifacts once per backend process and reuses them for subsequent inference requests.

---

# API Surface

The backend exposes endpoints for:

- Health checks
- Model information
- Integration contract
- Session creation
- Session inspection
- Audio chunk submission
- Session reset
- Session deletion
- WebSocket live-stream inference

The integration contract exposes the expected sample rate, chunk/window sizes, model version, transport paths, and enabled capabilities.

---

# Repository Structure

```text
real-time-voice-cloning-detection-security-pipeline/
│
├── backend/
│   ├── app/
│   │   ├── api/              # REST + WebSocket endpoints
│   │   ├── audio/            # Chunking, buffering, windows
│   │   ├── core/             # Backend core modules
│   │   ├── ml/
│   │   │   ├── connectors/   # ML model integration
│   │   │   ├── fusion.py     # Evidence fusion
│   │   │   ├── manager.py    # Model management
│   │   │   └── runtime.py    # Persistent ML runtime
│   │   ├── security/         # Security policy
│   │   ├── services/         # Detection services
│   │   ├── sessions/         # Streaming sessions
│   │   ├── temporal/         # Risk accumulation
│   │   └── transport/        # Telemetry
│   │
│   ├── tests/                # Backend tests
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── package-lock.json
│
├── scripts/                  # ML training, evaluation and inference
│
├── 00_PROJECT_DOCS/          # Dataset and project reference material
├── handoff/                  # Integration verification utilities
│
├── AGENTS.md                 # Project engineering conventions
├── ML_BACKEND_HANDOFF.md     # ML ↔ backend integration contract
└── README.md
```

---

# Technology Stack

## Machine Learning

The backend environment is pinned through `backend/requirements.txt`.

Core ML/audio dependencies include:

- Python
- PyTorch **2.11.0 + CUDA 12.8**
- TorchAudio
- TorchVision
- NumPy
- SciPy
- Librosa
- Scikit-learn
- Pandas
- SoundFile
- SoundDevice
- SoXR
- Joblib
- Matplotlib

## Backend

- FastAPI
- Uvicorn
- Pydantic
- WebSockets
- Python-dotenv
- PyYAML
- Requests
- Pytest

## Frontend

- React 19
- TypeScript
- Vite
- ESLint

The backend requirements file contains the pinned Python environment dependencies used by the project, including the supporting scientific-computing and runtime packages required by the ML and API layers.

---

# Integration Reference

The repository includes an ML/backend handoff layer for integrating the detector into another Python backend.

The integration material documents the expected:

- Audio input format
- Sample rate
- Window size
- Model version
- Model output
- Spoof probability
- Prosody evidence
- Quality/confidence information
- Risk score
- Risk level
- Runtime dependencies
- ML artifact locations
- Backend ↔ ML interface assumptions

The **ML ↔ backend handoff can therefore be used as a direct integration reference** when connecting the inference pipeline to another Python service.

---

# Local Setup

## Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The ML runtime requires the corresponding local frozen model and supporting evidence artifacts. These artifacts are intentionally excluded from the public repository.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Testing

Backend tests cover the main application layers, including:

- API health and session behavior
- Audio chunk buffering
- Exact 10-second window construction
- ML connector registration
- Model management
- Evidence-fusion behavior

Run backend tests with:

```bash
cd backend
pytest
```

---

# Project Engineering Principles

The implementation keeps several concerns deliberately separated:

```text
Audio Transport
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

This separation makes the inference contract explicit and allows the ML pipeline to be integrated without coupling model internals to transport or application-security logic.

---

# Limitations

This is a **personal ML engineering and research-oriented prototype**, not a certified voice-authentication system.

- Spoof detection does not establish speaker identity.
- Performance can vary across microphones, codecs, channels, environments, and unseen synthesis methods.
- Prosody is a secondary signal rather than the primary detector.
- Audio quality affects confidence but is not itself treated as spoof evidence.
- Production deployment would require broader external evaluation, calibration, monitoring, adversarial testing, privacy review, and security hardening.
- The reported benchmark results correspond to the documented evaluation and hardware configurations.

---

# Author

**Ankool Srivastava**

GitHub: [ankoolsrivastava](https://github.com/ankoolsrivastava)
