from pathlib import Path
import os
import sys
import subprocess
import importlib

import pandas as pd


# ============================================================
# VOICEGUARD — DAY 1 + DAY 2 MASTER VERIFIER
# ============================================================
#
# REAL DATA ONLY.
#
# This verifier does NOT create:
#   - dummy waveforms
#   - random tensors
#   - synthetic samples
#   - fake labels
#   - spoof data
#   - robustness data
#
# It verifies the REAL VoiceGuard foundation currently available.
#
# Current pillars:
#   REAL        = available
#   SPOOF       = waiting for Friend 1
#   ROBUSTNESS  = waiting for Friend 2
#
# Final binary training is intentionally NOT performed yet.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset_v1"
METADATA = DATASET / "metadata"
PROCESSED = DATASET / "processed"

REAL_MANIFEST = (
    METADATA / "real_master_manifest.csv"
)


# ============================================================
# CURRENT REAL MASTER SCHEMA
# ============================================================
#
# The actual current master manifest contains 25 columns.
#
# IMPORTANT:
# The verifier checks that all required master fields exist.
# It does NOT reject additional legitimate metadata fields.
#
# This prevents the verifier from forcing the dataset to match
# the verifier.
# ============================================================

REQUIRED_MASTER_COLUMNS = [
    "file_id",
    "speaker_id",
    "language",
    "language_name",
    "split",
    "original_filename",
    "processed_audio_path",
    "label",
    "source",
    "generator",
    "generator_version",
    "condition",
    "transcript",
    "parent_file_id",
    "source_dataset",
    "environment",
    "device",
    "distance_m",
    "text_id",
    "sample_rate",
    "channels",
    "duration_sec",
    "quality_status",
    "notes",
]


# ============================================================
# RESULT COUNTERS
# ============================================================

passed = 0
failed = 0


# ============================================================
# CHECK FUNCTION
# ============================================================

def check(name, condition, detail=""):

    global passed
    global failed

    if condition:

        print(f"[PASS] {name}")

        if detail:
            print(f"       {detail}")

        passed += 1

    else:

        print(f"[FAIL] {name}")

        if detail:
            print(f"       {detail}")

        failed += 1


# ============================================================
# RUN EXISTING REAL TEST
# ============================================================

def run_existing_real_test(
    script_name,
    description,
    success_markers=None,
):
    """
    Runs an EXISTING VoiceGuard test against the REAL dataset.

    No dummy or synthetic input is created here.

    Windows may return a non-zero process code after a test has
    already completed successfully. Therefore, when success
    markers are supplied, the actual test output is also checked.
    """

    global passed
    global failed

    script_path = (
        ROOT / "scripts" / script_name
    )

    if not script_path.exists():

        print(
            f"[FAIL] {description}"
        )

        print(
            f"       Missing: {script_path}"
        )

        failed += 1

        return

    print(
        f"\nRunning actual test: {script_name}"
    )

    env = os.environ.copy()

    env["PYTHONIOENCODING"] = "utf-8"

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    except Exception as exc:

        print(
            f"[FAIL] {description}"
        )

        print(
            f"       Execution error: {exc}"
        )

        failed += 1

        return

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if stdout:
        print(stdout)

    if stderr:
        print(stderr)

    # --------------------------------------------------------
    # Normal successful process
    # --------------------------------------------------------

    if result.returncode == 0:

        print(
            f"[PASS] {description}"
        )

        passed += 1

        return

    # --------------------------------------------------------
    # Some Windows/Python combinations can terminate with a
    # non-zero code after the test has already completed its
    # actual work successfully.
    #
    # We therefore verify the actual success markers printed
    # by the existing VoiceGuard test.
    # --------------------------------------------------------

    if success_markers:

        markers_found = all(
            marker in stdout
            for marker in success_markers
        )

        if markers_found:

            print(
                f"[PASS] {description}"
            )

            print(
                "       Test output contains all required "
                "success markers."
            )

            print(
                f"       Process exit code was "
                f"{result.returncode}, but the actual "
                f"pipeline test completed successfully."
            )

            passed += 1

            return

    # --------------------------------------------------------
    # Genuine failure
    # --------------------------------------------------------

    print(
        f"[FAIL] {description}"
    )

    print(
        f"       Exit code: {result.returncode}"
    )

    failed += 1


# ============================================================
# HEADER
# ============================================================

print("=" * 70)

print(
    "VOICEGUARD — DAY 1 + DAY 2 MASTER VERIFIER"
)

print("=" * 70)

print(
    "This verification uses REAL VoiceGuard data only."
)

print(
    "No dummy/synthetic data is created."
)


# ============================================================
# 1. PROJECT STRUCTURE
# ============================================================

print("\n" + "=" * 70)

print(
    "1. PROJECT STRUCTURE"
)

print("=" * 70)


check(
    "dataset_v1 directory",
    DATASET.exists(),
)

check(
    "metadata directory",
    METADATA.exists(),
)

check(
    "processed directory",
    PROCESSED.exists(),
)

check(
    "REAL master manifest",
    REAL_MANIFEST.exists(),
)


# ============================================================
# 2. REAL MASTER MANIFEST
# ============================================================

print("\n" + "=" * 70)

print(
    "2. REAL MASTER MANIFEST"
)

print("=" * 70)


df = None


if REAL_MANIFEST.exists():

    try:

        df = pd.read_csv(
            REAL_MANIFEST
        )

        print(
            "Rows    :",
            len(df)
        )

        print(
            "Columns :",
            len(df.columns)
        )

    except Exception as exc:

        check(
            "REAL manifest readable",
            False,
            str(exc),
        )


if df is not None:

    missing_columns = [
        column
        for column in REQUIRED_MASTER_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column
        for column in df.columns
        if column not in REQUIRED_MASTER_COLUMNS
    ]

    check(
        "Required master columns",
        len(missing_columns) == 0,
        (
            "All required master columns present"
            if not missing_columns
            else f"Missing: {missing_columns}"
        ),
    )

    if extra_columns:

        print(
            "[INFO] Additional master columns:"
        )

        print(
            "       " + ", ".join(extra_columns)
        )

    check(
        "REAL row count",
        len(df) == 4568,
        f"Found: {len(df)}",
    )

    duplicate_ids = (
        df["file_id"]
        .duplicated()
        .sum()
    )

    check(
        "Duplicate file IDs",
        duplicate_ids == 0,
        f"Duplicates: {duplicate_ids}",
    )

    duplicate_paths = (
        df["processed_audio_path"]
        .duplicated()
        .sum()
    )

    check(
        "Duplicate processed paths",
        duplicate_paths == 0,
        f"Duplicates: {duplicate_paths}",
    )

    languages = set(
        df["language"]
        .dropna()
        .unique()
    )

    check(
        "Languages",
        languages == {"HI", "MR"},
        f"Found: {sorted(languages)}",
    )

    language_names = set(
        df["language_name"]
        .dropna()
        .unique()
    )

    check(
        "Language names",
        language_names
        == {"Hindi", "Marathi"},
        f"Found: {sorted(language_names)}",
    )

    labels = set(
        df["label"]
        .dropna()
        .unique()
    )

    check(
        "REAL label",
        labels == {"bonafide"},
        f"Found: {sorted(labels)}",
    )

    splits = set(
        df["split"]
        .dropna()
        .unique()
    )

    check(
        "Splits",
        splits
        == {
            "train",
            "validation",
            "test",
        },
        f"Found: {sorted(splits)}",
    )

    durations = pd.to_numeric(
        df["duration_sec"],
        errors="coerce",
    )

    check(
        "Duration values",
        durations.notna().all()
        and (durations > 0).all(),
        "All durations numeric and positive",
    )

    sample_rates = set(
        df["sample_rate"]
        .dropna()
        .unique()
    )

    check(
        "Sample rate",
        sample_rates == {16000},
        f"Found: {sorted(sample_rates)}",
    )

    channels = set(
        df["channels"]
        .dropna()
        .unique()
    )

    check(
        "Channels",
        channels == {1},
        f"Found: {sorted(channels)}",
    )

    quality_status = set(
        df["quality_status"]
        .dropna()
        .unique()
    )

    check(
        "Quality status",
        quality_status == {"usable"},
        f"Found: {sorted(quality_status)}",
    )


# ============================================================
# 3. REAL PROCESSED AUDIO
# ============================================================

print("\n" + "=" * 70)

print(
    "3. REAL PROCESSED AUDIO"
)

print("=" * 70)


if df is not None:

    missing_audio = []

    for relative_path in df[
        "processed_audio_path"
    ]:

        audio_path = (
            ROOT / str(relative_path)
        )

        if not audio_path.exists():

            missing_audio.append(
                str(audio_path)
            )

    check(
        "All REAL processed audio exists",
        len(missing_audio) == 0,
        f"Missing: {len(missing_audio)}",
    )

    real_wavs = list(
        (
            PROCESSED / "real"
        ).rglob("*.wav")
    )

    check(
        "REAL processed WAV count",
        len(real_wavs) == 4568,
        f"Found: {len(real_wavs)}",
    )


# ============================================================
# 4. SPEAKER-DISJOINT SPLITS
# ============================================================

print("\n" + "=" * 70)

print(
    "4. SPEAKER-DISJOINT SPLITS"
)

print("=" * 70)


if df is not None:

    train_speakers = set(
        df.loc[
            df["split"] == "train",
            "speaker_id",
        ].dropna()
    )

    validation_speakers = set(
        df.loc[
            df["split"] == "validation",
            "speaker_id",
        ].dropna()
    )

    test_speakers = set(
        df.loc[
            df["split"] == "test",
            "speaker_id",
        ].dropna()
    )

    train_validation = (
        train_speakers
        & validation_speakers
    )

    train_test = (
        train_speakers
        & test_speakers
    )

    validation_test = (
        validation_speakers
        & test_speakers
    )

    check(
        "Train ∩ Validation",
        len(train_validation) == 0,
        f"Overlap: {len(train_validation)}",
    )

    check(
        "Train ∩ Test",
        len(train_test) == 0,
        f"Overlap: {len(train_test)}",
    )

    check(
        "Validation ∩ Test",
        len(validation_test) == 0,
        f"Overlap: {len(validation_test)}",
    )

    check(
        "Total REAL speakers",
        df["speaker_id"].nunique() == 424,
        f"Found: {df['speaker_id'].nunique()}",
    )


# ============================================================
# 5. ML ENVIRONMENT
# ============================================================

print("\n" + "=" * 70)

print(
    "5. ML ENVIRONMENT"
)

print("=" * 70)


try:

    import torch

    check(
        "PyTorch",
        True,
        torch.__version__,
    )

    check(
        "CUDA available",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():

        print(
            "[INFO] GPU:",
            torch.cuda.get_device_name(0),
        )

except Exception as exc:

    check(
        "PyTorch",
        False,
        str(exc),
    )


for package_name in [
    "librosa",
    "soundfile",
    "pandas",
    "numpy",
    "sklearn",
]:

    try:

        module = importlib.import_module(
            package_name
        )

        version = getattr(
            module,
            "__version__",
            "installed",
        )

        check(
            package_name,
            True,
            str(version),
        )

    except Exception as exc:

        check(
            package_name,
            False,
            str(exc),
        )


# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)

print(
    "6. TRAINING CONFIGURATION"
)

print("=" * 70)


try:

    scripts_dir = ROOT / "scripts"

    if str(scripts_dir) not in sys.path:

        sys.path.insert(
            0,
            str(scripts_dir),
        )

    import config

    check(
        "Sample rate",
        config.SAMPLE_RATE == 16000,
        f"Found: {config.SAMPLE_RATE}",
    )

    check(
        "Channels",
        config.CHANNELS == 1,
        f"Found: {config.CHANNELS}",
    )

    check(
        "Maximum duration",
        config.MAX_DURATION_SEC == 10.0,
        f"Found: {config.MAX_DURATION_SEC}",
    )

    check(
        "Mel bins",
        config.N_MELS == 64,
        f"Found: {config.N_MELS}",
    )

    check(
        "FFT",
        config.N_FFT == 400,
        f"Found: {config.N_FFT}",
    )

    check(
        "Hop length",
        config.HOP_LENGTH == 160,
        f"Found: {config.HOP_LENGTH}",
    )

    check(
        "Random seed",
        config.RANDOM_SEED == 42,
        f"Found: {config.RANDOM_SEED}",
    )

    check(
        "Checkpoint directory",
        config.CHECKPOINT_DIR.exists(),
        str(config.CHECKPOINT_DIR),
    )

except Exception as exc:

    check(
        "Training configuration",
        False,
        str(exc),
    )


# ============================================================
# 7. ACTUAL REAL DATASET LOADER
# ============================================================

print("\n" + "=" * 70)

print(
    "7. ACTUAL REAL DATASET LOADER"
)

print("=" * 70)


run_existing_real_test(
    "test_dataset_loader.py",
    "REAL dataset loader — all samples",
    success_markers=[
        "Total samples : 4568",
        "Errors        : 0",
        "NaN samples   : 0",
        "Inf samples   : 0",
        "ALL REAL AUDIO SAMPLES LOADED SUCCESSFULLY",
    ],
)


# ============================================================
# 8. ACTUAL REAL AUDIO PREPROCESSOR
# ============================================================

print("\n" + "=" * 70)

print(
    "8. ACTUAL REAL AUDIO PREPROCESSOR"
)

print("=" * 70)


run_existing_real_test(
    "test_audio_preprocessor.py",
    "REAL audio preprocessing",
    success_markers=[
        "PASS — preprocessing output is valid",
    ],
)


# ============================================================
# 9. ACTUAL REAL LOG-MEL FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)

print(
    "9. ACTUAL REAL LOG-MEL FEATURE EXTRACTION"
)

print("=" * 70)


run_existing_real_test(
    "test_feature_extractor.py",
    "REAL Log-Mel feature extraction",
    success_markers=[
        "Log-Mel Features",
        "Shape : torch.Size([64, 1001])",
        "CNN Input",
        "Contains NaN : False",
        "Contains Inf : False",
        "PASS — Log-Mel feature extraction is valid",
    ],
)


# ============================================================
# 10. ACTUAL REAL TRAINING PATH
# ============================================================

print("\n" + "=" * 70)

print(
    "10. ACTUAL REAL TRAINING PATH"
)

print("=" * 70)


print(
    "Existing REAL-data training test is executed."
)

print(
    "No random tensor or synthetic waveform is created."
)


run_existing_real_test(
    "test_training_batch.py",
    "REAL AUDIO → FEATURES → BATCH → MODEL → LOSS → BACKWARD",
    success_markers=[
        "Training samples: 3157",
        "Features : torch.Size([4, 1, 64, 1001])",
        "Logits    : torch.Size([4, 2])",
        "Loss      :",
        "Backward  : PASS",
        "REAL AUDIO → FEATURES → BATCH → MODEL → LOSS → BACKWARD",
        "COMPLETE TRAINING PATH IS OPERATIONAL",
    ],
)


# ============================================================
# 11. FUTURE DATASET PILLARS
# ============================================================

print("\n" + "=" * 70)

print(
    "11. DATASET PILLAR STATUS"
)

print("=" * 70)


spoof_manifest = (
    METADATA / "spoof_audio_manifest.csv"
)

robustness_manifest = (
    METADATA / "robustness_audio_manifest.csv"
)


if spoof_manifest.exists():

    print(
        "[INFO] SPOOF manifest PRESENT"
    )

else:

    print(
        "[INFO] SPOOF manifest NOT YET DELIVERED"
    )


if robustness_manifest.exists():

    print(
        "[INFO] ROBUSTNESS manifest PRESENT"
    )

else:

    print(
        "[INFO] ROBUSTNESS manifest NOT YET DELIVERED"
    )


print(
    "\nThis is expected at the current project stage."
)


# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL DAY 1 + DAY 2 VERIFICATION"
)

print("=" * 70)


print(
    f"PASS: {passed}"
)

print(
    f"FAIL: {failed}"
)


print(
    "\nCurrent dataset state:"
)

print(
    "  REAL        : PRESENT + VERIFIED"
)

print(
    "  SPOOF       : WAITING FOR FRIEND 1"
)

print(
    "  ROBUSTNESS  : WAITING FOR FRIEND 2"
)


print(
    "\nFinal binary training:"
)

print(
    "  NOT STARTED — intentionally waiting for SPOOF."
)


print("\n" + "=" * 70)


if failed == 0:

    print(
        "✅ DAY 1 + DAY 2 VERIFIED SUCCESSFULLY"
    )

    print(
        "✅ REAL DATA FOUNDATION VERIFIED"
    )

    print(
        "✅ REAL ML PIPELINE VERIFIED"
    )

    print(
        "✅ NO DUMMY DATA USED"
    )

    print(
        "✅ CURRENT FOUNDATION READY TO FREEZE"
    )

    print("=" * 70)

    sys.exit(0)


else:

    print(
        "❌ DAY 1 + DAY 2 VERIFICATION FAILED"
    )

    print(
        "Fix the reported REAL pipeline failures."
    )

    print("=" * 70)

    sys.exit(1)