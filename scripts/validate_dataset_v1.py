from pathlib import Path
import pandas as pd
import wave
import sys


ROOT = Path("dataset_v1")

REAL_MANIFEST = ROOT / "metadata" / "real_master_manifest.csv"
SPOOF_MANIFEST = ROOT / "metadata" / "spoof_audio_manifest.csv"
ROBUSTNESS_MANIFEST = ROOT / "metadata" / "robustness_audio_manifest.csv"


MASTER_COLUMNS = [
    "file_id",
    "parent_file_id",
    "speaker_id",
    "language",
    "language_name",
    "split",
    "label",
    "source",
    "source_dataset",
    "original_filename",
    "original_source_path",
    "processed_audio_path",
    "generator",
    "generator_version",
    "condition",
    "environment",
    "device",
    "distance_m",
    "transcript",
    "text_id",
    "duration_sec",
    "sample_rate",
    "channels",
    "quality_status",
    "notes",
]

VALID_LANGUAGES = {
    "HI": "Hindi",
    "MR": "Marathi",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}

VALID_LABELS = {
    "bonafide",
    "spoof",
}

VALID_QUALITY = {
    "usable",
    "excluded_duration",
    "excluded",
    "invalid",
}


def check_master_columns(df, name):
    missing = [c for c in MASTER_COLUMNS if c not in df.columns]

    if missing:
        print(f"FAIL — {name} missing columns:")
        for c in missing:
            print("  -", c)
        return False

    extra = [c for c in df.columns if c not in MASTER_COLUMNS]

    if extra:
        print(f"WARNING — {name} has extra columns:")
        for c in extra:
            print("  -", c)

    print(f"PASS — {name} has all master columns")
    return True


def validate_manifest(path, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if not path.exists():
        print("INFO — manifest not present:")
        print(path)
        return None

    df = pd.read_csv(path)

    print("Path    :", path)
    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    critical_error = False

    # --------------------------------------------------------
    # 1. MASTER COLUMNS
    # --------------------------------------------------------

    print("\n[1] MASTER COLUMNS")

    if not check_master_columns(df, name):
        critical_error = True

    # --------------------------------------------------------
    # 2. FILE IDS
    # --------------------------------------------------------

    print("\n[2] FILE IDs")

    duplicates = df["file_id"].duplicated().sum()

    if duplicates == 0:
        print("PASS — no duplicate file IDs")
    else:
        print("FAIL — duplicate file IDs:", duplicates)
        critical_error = True

    # --------------------------------------------------------
    # 3. LANGUAGES
    # --------------------------------------------------------

    print("\n[3] LANGUAGES")

    languages = sorted(df["language"].dropna().unique())

    print("Found:", languages)

    invalid_languages = [
        x for x in languages
        if x not in VALID_LANGUAGES
    ]

    if invalid_languages:
        print("FAIL — invalid languages:", invalid_languages)
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 4. LANGUAGE NAME CONSISTENCY
    # --------------------------------------------------------

    print("\n[4] LANGUAGE NAME CONSISTENCY")

    mismatches = []

    for row in df.itertuples(index=False):
        if row.language in VALID_LANGUAGES:
            expected = VALID_LANGUAGES[row.language]

            if str(row.language_name) != expected:
                mismatches.append(
                    (row.file_id, row.language, row.language_name)
                )

    if mismatches:
        print("FAIL — mismatches:", len(mismatches))
        for x in mismatches[:10]:
            print(" ", x)
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 5. LABELS
    # --------------------------------------------------------

    print("\n[5] LABELS")

    labels = sorted(df["label"].dropna().unique())

    print("Found:", labels)

    invalid_labels = [
        x for x in labels
        if x not in VALID_LABELS
    ]

    if invalid_labels:
        print("FAIL — invalid labels:", invalid_labels)
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 6. SPLITS
    # --------------------------------------------------------

    print("\n[6] SPLITS")

    splits = sorted(df["split"].dropna().unique())

    print("Found:", splits)

    invalid_splits = [
        x for x in splits
        if x not in VALID_SPLITS
    ]

    if invalid_splits:
        print("FAIL — invalid splits:", invalid_splits)
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 7. DURATION
    # --------------------------------------------------------

    print("\n[7] DURATION")

    duration = pd.to_numeric(
        df["duration_sec"],
        errors="coerce"
    )

    missing_duration = duration.isna().sum()
    non_positive = (duration <= 0).sum()

    print("Missing/non-numeric:", missing_duration)
    print("Non-positive        :", non_positive)

    if missing_duration or non_positive:
        print("FAIL")
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 8. SAMPLE RATE
    # --------------------------------------------------------

    print("\n[8] SAMPLE RATE")

    sample_rates = sorted(
        pd.to_numeric(
            df["sample_rate"],
            errors="coerce"
        ).dropna().unique()
    )

    print("Found:", sample_rates)

    invalid_sr = [
        x for x in sample_rates
        if x != 16000
    ]

    if invalid_sr:
        print("FAIL — expected 16000 Hz")
        critical_error = True
    else:
        print("PASS — 16000 Hz")

    # --------------------------------------------------------
    # 9. CHANNELS
    # --------------------------------------------------------

    print("\n[9] CHANNELS")

    channels = sorted(
        pd.to_numeric(
            df["channels"],
            errors="coerce"
        ).dropna().unique()
    )

    print("Found:", channels)

    invalid_channels = [
        x for x in channels
        if x != 1
    ]

    if invalid_channels:
        print("FAIL — expected mono")
        critical_error = True
    else:
        print("PASS — mono")

    # --------------------------------------------------------
    # 10. QUALITY
    # --------------------------------------------------------

    print("\n[10] QUALITY STATUS")

    qualities = sorted(
        df["quality_status"].dropna().unique()
    )

    print("Found:", qualities)

    invalid_quality = [
        x for x in qualities
        if x not in VALID_QUALITY
    ]

    if invalid_quality:
        print("FAIL — invalid quality values")
        critical_error = True
    else:
        print("PASS")

    # --------------------------------------------------------
    # 11. PROCESSED AUDIO PATHS
    # --------------------------------------------------------

    print("\n[11] PROCESSED AUDIO PATHS")

    missing_audio = []
    duplicate_paths = df["processed_audio_path"].duplicated().sum()

    for row in df.itertuples(index=False):

        path = Path(str(row.processed_audio_path))

        if not path.exists():
            missing_audio.append(
                (row.file_id, str(path))
            )

    print("Missing audio:", len(missing_audio))
    print("Duplicate paths:", duplicate_paths)

    if missing_audio:
        print("FAIL — missing processed audio")
        for x in missing_audio[:10]:
            print(" ", x)
        critical_error = True
    else:
        print("PASS — all processed audio exists")

    if duplicate_paths:
        print("FAIL — duplicate processed paths")
        critical_error = True
    else:
        print("PASS — no duplicate processed paths")

    # --------------------------------------------------------
    # 12. PARENT LINEAGE
    # --------------------------------------------------------

    print("\n[12] PARENT LINEAGE")

    parent_missing = 0

    for value in df["parent_file_id"]:
        if pd.isna(value) or str(value).strip() == "":
            parent_missing += 1

    print("NA parent IDs:", parent_missing)

    # For REAL data, NA is expected.
    if name == "REAL MASTER MANIFEST":
        print("INFO — NA parent IDs expected for original REAL audio")

    # --------------------------------------------------------
    # 13. REAL LABEL SANITY
    # --------------------------------------------------------

    if name == "REAL MASTER MANIFEST":

        print("\n[13] REAL LABEL SANITY")

        spoof_count = (
            df["label"] == "spoof"
        ).sum()

        print("Spoof rows:", spoof_count)

        if spoof_count != 0:
            print("FAIL — REAL manifest contains spoof rows")
            critical_error = True
        else:
            print("PASS — REAL contains only bonafide")

    # --------------------------------------------------------
    # 14. DISTRIBUTIONS
    # --------------------------------------------------------

    print("\n[14] LANGUAGE DISTRIBUTION")

    print(
        df["language_name"]
        .value_counts()
        .to_string()
    )

    print("\n[15] LABEL DISTRIBUTION")

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    print("\n[16] SPLIT DISTRIBUTION")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    print("\n[17] QUALITY DISTRIBUTION")

    print(
        df["quality_status"]
        .value_counts()
        .to_string()
    )

    return {
        "df": df,
        "critical_error": critical_error,
    }


def main():

    print("=" * 70)
    print("VOICEGUARD — DATASET V1 MASTER VALIDATION")
    print("=" * 70)

    results = {}

    results["real"] = validate_manifest(
        REAL_MANIFEST,
        "REAL MASTER MANIFEST"
    )

    results["spoof"] = validate_manifest(
        SPOOF_MANIFEST,
        "SPOOF MANIFEST"
    )

    results["robustness"] = validate_manifest(
        ROBUSTNESS_MANIFEST,
        "ROBUSTNESS MANIFEST"
    )

    # --------------------------------------------------------
    # CROSS-PILLAR CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-PILLAR VALIDATION")
    print("=" * 70)

    manifests = []

    for result in results.values():

        if result is not None:
            manifests.append(result["df"])

    if manifests:

        combined_ids = pd.concat(
            [x["file_id"] for x in manifests],
            ignore_index=True
        )

        duplicate_global_ids = (
            combined_ids.duplicated().sum()
        )

        print(
            "\nGlobal duplicate file IDs:",
            duplicate_global_ids
        )

        if duplicate_global_ids == 0:
            print("PASS")
        else:
            print("FAIL")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    errors = [
        x["critical_error"]
        for x in results.values()
        if x is not None
    ]

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if any(errors):
        print("❌ FAIL — critical validation errors found")
        sys.exit(1)

    print("✅ PASS — no critical errors found")


if __name__ == "__main__":
    main()