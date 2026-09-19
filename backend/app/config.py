from pathlib import Path
import json
import os

import torch


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

ML_HANDOFF_ROOT = Path(
    os.getenv(
        "VOICEGUARD_ML_HANDOFF_ROOT",
        str(PROJECT_ROOT / "ml_handoff"),
    )
).resolve()

MODEL_PATH = ML_HANDOFF_ROOT / "models" / "best_model.pt"
PROSODY_SCORER_PATH = ML_HANDOFF_ROOT / "prosody" / "prosody_scorer_v1.pkl"
PROSODY_CONFIG_PATH = ML_HANDOFF_ROOT / "prosody" / "PROSODY_FUSION_CONFIG_V1.json"

ML_RUNTIME_ROOT = ML_HANDOFF_ROOT / "runtime"
ML_SCRIPTS_ROOT = ML_RUNTIME_ROOT / "scripts"

SAMPLE_RATE = 16000
WINDOW_DURATION_SEC = 10.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_SEC)

N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64
FMIN = 20
FMAX = 8000

NUM_CLASSES = 2
HIDDEN_SIZE = 128
SPOOF_THRESHOLD = 0.25

DEVICE = os.getenv(
    "VOICEGUARD_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


def load_prosody_config() -> dict:
    with PROSODY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_paths() -> None:
    required = {
        "ML handoff": ML_HANDOFF_ROOT,
        "model": MODEL_PATH,
        "prosody scorer": PROSODY_SCORER_PATH,
        "prosody config": PROSODY_CONFIG_PATH,
        "ML runtime": ML_RUNTIME_ROOT,
        "ML scripts": ML_SCRIPTS_ROOT,
    }

    missing = {
        name: str(path)
        for name, path in required.items()
        if not path.exists()
    }

    if missing:
        details = "\n".join(
            f"  - {name}: {path}"
            for name, path in missing.items()
        )
        raise FileNotFoundError(
            "VoiceGuard ML runtime prerequisites are missing:\n" + details
        )
