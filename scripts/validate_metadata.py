import pandas as pd
from pathlib import Path

# ============================================================
# VoiceGuard — Dataset V1 Metadata Validator
# Read-only audit: does NOT modify any dataset files.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = PROJECT_ROOT / "dataset_v1" / "metadata" / "real_audio_manifest.csv"

REQUIRED_COLUMNS = [
    "file_id",
    "speaker_id",
    "language",
    "language_name",
    "split",
    "original_filename",
    "source_path",
    "label",
    "source",
    "generator",
    "generator_version",
    "condition",
    "transcript",
    "age",
    "gender",
    "accents",
    "client_id",
    "duration_sec",
    "quality_status",
]

VALID_LANGUAGES = {
    "HI": "Hindi",
    "MR": "Marathi",
}

VALID_SPLITS = {"train", "validation", "test"}
VALID_LABELS = {"bonafide", "spoof"}
VALID_STATUS = {"usable", "excluded_duration", "excluded", "invalid"}

print("=" * 65)
print("VOICEGUARD — DATASET V1 METADATA VALIDATION")
print("=" * 65)

if not MANIFEST.exists():
    print(f"\nERROR: Manifest not found:")
    print(MANIFEST)
    raise SystemExit(1)

df = pd.read_csv(MANIFEST)

errors = []
warnings = []

# ------------------------------------------------------------
# 1. Basic information
# ------------------------------------------------------------

print("\n[1] BASIC INFORMATION")
print(f"Manifest : {MANIFEST}")
print(f"Rows     : {len(df)}")
print(f"Columns  : {len(df.columns)}")

# ------------------------------------------------------------
# 2. Required columns
# ------------------------------------------------------------

print("\n[2] REQUIRED COLUMNS")

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    errors.append(
        f"Missing required columns: {missing_columns}"
    )
    print("FAIL")
    print("Missing:", missing_columns)
else:
    print("PASS — all current REAL manifest columns present")

# ------------------------------------------------------------
# 3. Duplicate file IDs
# ------------------------------------------------------------

print("\n[3] DUPLICATE FILE IDs")

if "file_id" in df.columns:
    duplicate_ids = df["file_id"].duplicated(keep=False)
    count = int(duplicate_ids.sum())

    if count:
        errors.append(f"Duplicate file_id rows: {count}")
        print(f"FAIL — {count} duplicate rows")
    else:
        print("PASS — 0 duplicates")

# ------------------------------------------------------------
# 4. Duplicate source paths
# ------------------------------------------------------------

print("\n[4] DUPLICATE SOURCE PATHS")

if "source_path" in df.columns:
    duplicate_paths = df["source_path"].duplicated(keep=False)
    count = int(duplicate_paths.sum())

    if count:
        warnings.append(
            f"Duplicate source_path rows: {count}"
        )
        print(f"WARNING — {count} duplicate rows")
    else:
        print("PASS — 0 duplicates")

# ------------------------------------------------------------
# 5. Labels
# ------------------------------------------------------------

print("\n[5] LABELS")

if "label" in df.columns:
    values = set(df["label"].dropna().astype(str).unique())
    invalid = values - VALID_LABELS

    print("Found:", sorted(values))

    if invalid:
        errors.append(f"Invalid labels: {sorted(invalid)}")
        print("FAIL — invalid:", sorted(invalid))
    else:
        print("PASS")

# ------------------------------------------------------------
# 6. Languages
# ------------------------------------------------------------

print("\n[6] LANGUAGES")

if "language" in df.columns:
    values = set(df["language"].dropna().astype(str).unique())
    invalid = values - set(VALID_LANGUAGES)

    print("Found:", sorted(values))

    if invalid:
        errors.append(f"Invalid language codes: {sorted(invalid)}")
        print("FAIL — invalid:", sorted(invalid))
    else:
        print("PASS")

# ------------------------------------------------------------
# 7. Language names match codes
# ------------------------------------------------------------

print("\n[7] LANGUAGE CODE ↔ NAME CONSISTENCY")

if "language" in df.columns and "language_name" in df.columns:

    mismatches = df[
        df["language"].notna()
        & df["language_name"].notna()
        & (
            df["language_name"]
            != df["language"].map(VALID_LANGUAGES)
        )
    ]

    if len(mismatches):
        errors.append(
            f"Language/name mismatches: {len(mismatches)}"
        )
        print(f"FAIL — {len(mismatches)} mismatches")
    else:
        print("PASS")

# ------------------------------------------------------------
# 8. Splits
# ------------------------------------------------------------

print("\n[8] SPLITS")

if "split" in df.columns:
    values = set(df["split"].dropna().astype(str).unique())
    invalid = values - VALID_SPLITS

    print("Found:", sorted(values))

    if invalid:
        errors.append(f"Invalid splits: {sorted(invalid)}")
        print("FAIL — invalid:", sorted(invalid))
    else:
        print("PASS")

# ------------------------------------------------------------
# 9. Duration
# ------------------------------------------------------------

print("\n[9] DURATION")

if "duration_sec" in df.columns:

    duration = pd.to_numeric(
        df["duration_sec"],
        errors="coerce"
    )

    missing = int(duration.isna().sum())
    non_positive = int((duration <= 0).sum())

    print(f"Missing/non-numeric: {missing}")
    print(f"Non-positive        : {non_positive}")

    if missing:
        errors.append(
            f"Missing/non-numeric durations: {missing}"
        )

    if non_positive:
        errors.append(
            f"Non-positive durations: {non_positive}"
        )

    if missing == 0 and non_positive == 0:
        print("PASS")

# ------------------------------------------------------------
# 10. Quality status
# ------------------------------------------------------------

print("\n[10] QUALITY STATUS")

if "quality_status" in df.columns:
    values = set(
        df["quality_status"]
        .dropna()
        .astype(str)
        .unique()
    )

    invalid = values - VALID_STATUS

    print("Found:", sorted(values))

    if invalid:
        warnings.append(
            f"Unrecognized quality_status values: {sorted(invalid)}"
        )
        print("WARNING — unrecognized:", sorted(invalid))
    else:
        print("PASS")

# ------------------------------------------------------------
# 11. Missing values
# ------------------------------------------------------------

print("\n[11] MISSING VALUES")

missing = df.isna().sum()
missing = missing[missing > 0]

if len(missing):
    print(missing.to_string())
    warnings.append(
        "Some metadata fields contain missing values."
    )
else:
    print("PASS — no missing values")

# ------------------------------------------------------------
# 12. REAL dataset-specific checks
# ------------------------------------------------------------

print("\n[12] REAL DATASET CHECKS")

if "label" in df.columns:
    spoof_rows = int((df["label"] == "spoof").sum())
    bonafide_rows = int((df["label"] == "bonafide").sum())

    print(f"Bonafide rows: {bonafide_rows}")
    print(f"Spoof rows   : {spoof_rows}")

    if spoof_rows == 0:
        print("INFO — current REAL manifest contains no spoof rows.")

# ------------------------------------------------------------
# 13. Usable rows
# ------------------------------------------------------------

print("\n[13] QUALITY SUMMARY")

if "quality_status" in df.columns:
    print(
        df["quality_status"]
        .value_counts(dropna=False)
        .to_string()
    )

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL VALIDATION RESULT")
print("=" * 65)

if errors:
    print("\n❌ FAIL")
    print("\nErrors:")
    for error in errors:
        print(" -", error)
else:
    print("\n✅ NO CRITICAL METADATA ERRORS FOUND")

if warnings:
    print("\nWarnings:")
    for warning in warnings:
        print(" -", warning)

print("\nThis validator is READ-ONLY.")
print("No dataset files were modified.")
