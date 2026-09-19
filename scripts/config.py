from pathlib import Path


# ==========================================================
# VOICEGUARD — CENTRAL CONFIGURATION
# ==========================================================

# ----------------------------------------------------------
# Project paths
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset_v1"

METADATA_DIR = DATASET_ROOT / "metadata"

PROCESSED_DIR = DATASET_ROOT / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR = MODELS_DIR / "checkpoints"


# ----------------------------------------------------------
# Canonical Master Dataset V1
# ----------------------------------------------------------

MASTER_DATASET_V1_DIR = (
    PROJECT_ROOT / "master_v1"
)

MASTER_MANIFEST = (
    MASTER_DATASET_V1_DIR /
    "metadata" /
    "master_dataset_v1.csv"
)

MASTER_LOCK = (
    MASTER_DATASET_V1_DIR /
    "reports" /
    "MASTER_DATASET_V1_LOCK.json"
)


# ----------------------------------------------------------
# Audio configuration
# ----------------------------------------------------------

SAMPLE_RATE = 16000

CHANNELS = 1

MAX_DURATION_SEC = 10.0

TARGET_SAMPLES = int(
    SAMPLE_RATE * MAX_DURATION_SEC
)


# ----------------------------------------------------------
# Log-Mel configuration
# ----------------------------------------------------------

N_FFT = 400

HOP_LENGTH = 160

WIN_LENGTH = 400

N_MELS = 64

FMIN = 20

FMAX = 8000


# ----------------------------------------------------------
# Model configuration
# ----------------------------------------------------------

NUM_CLASSES = 2

GRU_HIDDEN_SIZE = 128


# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

BATCH_SIZE = 4

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

EPOCHS = 30

EARLY_STOPPING_PATIENCE = 7


# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------

RANDOM_SEED = 42


# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}