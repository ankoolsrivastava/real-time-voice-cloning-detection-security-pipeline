# Real-Time Voice Anti-Spoofing & Impersonation Detection

A real-time / near-live voice security pipeline for detecting **bonafide vs spoofed speech** and converting model evidence into an operational impersonation risk score.

The system combines a frozen deep-learning detector with secondary prosodic evidence, audio-quality telemetry, temporal aggregation, and a policy layer exposed through REST and WebSocket interfaces.

> **Project type:** Personal ML engineering project  
> **Scope:** Hindi and Marathi speech  
> **Status:** Working prototype with an integrated backend and frontend

## What it does

The pipeline analyzes incoming speech in short windows and identifies signals associated with synthetic, cloned, or replayed audio.

```
Microphone / Audio Stream
        ↓
Chunk Buffering & Windowing
        ↓
16 kHz Mono Preprocessing
        ↓
Log-Mel Spectrogram
        ↓
CNN + BiGRU + Temporal Attention
        ↓
Spoof Evidence
        ├──────────────┐
        ↓              ↓
Prosody Evidence   Audio Quality
        └───────┬──────┘
                ↓
        Evidence Fusion
                ↓
      Temporal Risk Accumulation
                ↓
       0–100 Risk Score
                ↓
 Security Policy / API Response
```

## Key engineering components

### ML inference

- CNN + BiGRU + temporal-attention detector
- 16 kHz mono audio processing
- Fixed 10-second inference windows
- 64-bin log-Mel spectrogram features
- BONAFIDE / SPOOF classification
- Explicit preprocessing, thresholding, and model contract

### Evidence fusion

The primary detector is combined with a secondary prosody scorer.

Current fusion configuration:

- **V2 detector:** 83%
- **Prosody evidence:** 17%

Audio-quality measurements are treated as **confidence/reliability telemetry**, not direct spoof evidence.

### Streaming backend

The backend provides:

- audio chunk ingestion and buffering
- exact inference-window management
- timestamped processing
- session management
- temporal risk accumulation
- REST endpoints
- WebSocket streaming
- structured telemetry
- configurable security policy
- automated tests

### Risk engine

Detection evidence is converted into a **0–100 impersonation risk score** with four policy bands:

| Score | Level |
|---|---|
| 0–24 | LOW |
| 25–49 | MEDIUM |
| 50–74 | HIGH |
| 75–100 | CRITICAL |

The risk layer is designed to support downstream verification or escalation. It is not an identity oracle.

## Model evaluation

The frozen detector was evaluated on held-out data and robustness conditions.

| Evaluation | Result |
|---|---:|
| Clean test accuracy @ 0.50 | 97.76% |
| Clean test ROC-AUC | 99.69% |
| Clean test EER | 1.98% |
| Common 10K accuracy @ 0.25 | 96.34% |
| Common 10K F1 @ 0.25 | 96.40% |
| Dual-RIR accuracy @ 0.25 | 92.90% |
| Dual-RIR F1 @ 0.25 | 93.00% |

RTX 3050 inference benchmarking measured approximately **13.06 ms mean end-to-end latency** and **3.46 ms model-only latency** under the documented benchmark configuration.

These figures apply to the specified evaluation/benchmark setups and are not universal production guarantees.

## Dataset

The training and evaluation pipeline uses a locked internal dataset containing:

- bonafide speech
- clean synthetic speech
- real speech under robustness conditions
- spoof speech under robustness conditions
- Hindi and Marathi samples
- speaker-aware splits
- quality-control validation

Raw audio, protected test material, and local model artifacts are intentionally **not committed to this repository**.

## Repository structure

```
.
├── backend/                 # Streaming inference API and runtime
│   ├── app/
│   │   ├── api/             # REST + WebSocket interfaces
│   │   ├── audio/           # Chunking and window management
│   │   ├── ml/              # Model connectors and evidence fusion
│   │   ├── security/        # Security policy
│   │   ├── services/        # Detection service
│   │   ├── sessions/        # Session state
│   │   ├── temporal/        # Temporal aggregation
│   │   └── transport/       # Telemetry
│   └── tests/
│
├── frontend/                # React + TypeScript web interface
├── scripts/                 # Dataset, feature, training and evaluation utilities
├── handoff/                 # ML/backend integration records
└── README.md
```

## Tech stack

**Machine Learning**  
Python · PyTorch · NumPy · librosa · scikit-learn

**Backend**  
Python · FastAPI · WebSockets · Pydantic

**Frontend**  
React · TypeScript · Vite

**Engineering**  
Modular architecture · automated testing · reproducible preprocessing · model contracts · inference benchmarking

## Running locally

From the `backend/` directory:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The complete inference runtime requires the locally stored model/evidence artifacts referenced by the ML configuration. Those artifacts are intentionally excluded from Git.

## Repository hygiene

This repository intentionally excludes:

- raw audio datasets and generated speech corpora
- protected test data
- model checkpoints and large binary artifacts
- virtual environments and caches
- secrets and credentials
- machine-specific absolute paths

See `.gitignore` for repository-level exclusions.

## Limitations

This is a personal engineering and research-oriented prototype, not a certified voice-authentication or identity-verification system.

- Detection performance depends on the acoustic/channel conditions represented during evaluation.
- Model output represents spoofing evidence, not proof of speaker identity.
- Prosody is secondary evidence.
- Audio quality is used for confidence/reliability, not as spoof evidence.
- Broader external evaluation, calibration, monitoring, and security review would be required for production deployment.

## Author

**Ankool Srivastava**

[GitHub](https://github.com/ankoolsrivastava)

---

This repository focuses on the implemented ML inference pipeline, streaming backend, frontend integration, evaluation methodology, and engineering decisions while keeping private datasets and machine-specific artifacts outside version control.
