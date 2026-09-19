from pathlib import Path
import pandas as pd
import wave
import sys


# ============================================================
# VOICEGUARD — BUILD REAL MASTER MANIFEST
# ============================================================

ROOT = Path("dataset_v1")

INPUT_MANIFEST = ROOT / "metadata" / "real_audio_manifest.csv"
OUTPUT_MANIFEST = ROOT / "metadata" / "real_master_manifest.csv"

PROCESSED_ROOT = ROOT / "processed" / "real"


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


LANGUAGE_FOLDER = {
    "HI": "hindi",
    "MR": "marathi",
}


def get_audio_info(path):
    """Read WAV properties."""
    with wave.open(str(path), "rb") as wav:
        return {
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "frames": wav.getnframes(),
            "duration": wav.getnframes() / wav.getframerate(),
        }


def clean_value(value):
    """Convert pandas NaN to NA."""
    if pd.isna(value):
        return "NA"
    return value


def main():

    print("=" * 70)
    print("VOICEGUARD — BUILD REAL MASTER MANIFEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_MANIFEST.exists():
        print(f"ERROR — Input manifest not found:")
        print(INPUT_MANIFEST)
        sys.exit(1)

    df = pd.read_csv(INPUT_MANIFEST)

    print(f"\nInput rows: {len(df)}")

    # --------------------------------------------------------
    # Keep usable audio only
    # --------------------------------------------------------

    usable = df[df["quality_status"] == "usable"].copy()

    print(f"Usable rows: {len(usable)}")

    # --------------------------------------------------------
    # Build master records
    # --------------------------------------------------------

    records = []

    errors = []

    for row in usable.itertuples(index=False):

        file_id = row.file_id
        language = row.language

        if language not in LANGUAGE_FOLDER:
            errors.append(
                f"{file_id}: unsupported language {language}"
            )
            continue

        language_folder = LANGUAGE_FOLDER[language]

        processed_path = (
            PROCESSED_ROOT
            / language_folder
            / f"{file_id}.wav"
        )

        # ----------------------------------------------------
        # Verify processed WAV exists
        # ----------------------------------------------------

        if not processed_path.exists():
            errors.append(
                f"{file_id}: processed WAV missing: {processed_path}"
            )
            continue

        # ----------------------------------------------------
        # Verify WAV properties
        # ----------------------------------------------------

        try:
            audio = get_audio_info(processed_path)

        except Exception as exc:
            errors.append(
                f"{file_id}: unable to read WAV: {exc}"
            )
            continue

        if audio["sample_rate"] != 16000:
            errors.append(
                f"{file_id}: sample rate is "
                f"{audio['sample_rate']}, expected 16000"
            )

        if audio["channels"] != 1:
            errors.append(
                f"{file_id}: channels is "
                f"{audio['channels']}, expected 1"
            )

        if audio["sample_width"] != 2:
            errors.append(
                f"{file_id}: bit depth is "
                f"{audio['sample_width'] * 8}, expected 16"
            )

        if audio["duration"] <= 0:
            errors.append(
                f"{file_id}: invalid duration"
            )

        # ----------------------------------------------------
        # Create master record
        # ----------------------------------------------------

        record = {
            "file_id": file_id,
            "parent_file_id": "NA",

            "speaker_id": clean_value(row.speaker_id),

            "language": language,
            "language_name": clean_value(row.language_name),

            "split": clean_value(row.split),

            "label": clean_value(row.label),

            "source": clean_value(row.source),
            "source_dataset": clean_value(row.source),

            "original_filename": clean_value(
                row.original_filename
            ),

            "original_source_path": clean_value(
                row.source_path
            ),

            "processed_audio_path": str(
                processed_path
            ).replace("\\", "/"),

            "generator": clean_value(row.generator),
            "generator_version": clean_value(
                row.generator_version
            ),

            "condition": clean_value(row.condition),

            "environment": "NA",
            "device": "NA",
            "distance_m": "NA",

            "transcript": clean_value(row.transcript),

            "text_id": "NA",

            "duration_sec": round(
                audio["duration"], 3
            ),

            "sample_rate": audio["sample_rate"],
            "channels": audio["channels"],

            "quality_status": "usable",

            "notes": "Derived from validated REAL manifest",
        }

        records.append(record)

    # --------------------------------------------------------
    # Stop if errors
    # --------------------------------------------------------

    if errors:

        print("\n" + "=" * 70)
        print("ERRORS FOUND")
        print("=" * 70)

        for error in errors[:50]:
            print(" -", error)

        if len(errors) > 50:
            print(
                f"... and {len(errors) - 50} more errors"
            )

        print(
            "\nMaster manifest was NOT written because "
            "audio validation errors were found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    master = pd.DataFrame(records)

    master = master[MASTER_COLUMNS]

    # --------------------------------------------------------
    # Global checks
    # --------------------------------------------------------

    duplicate_ids = master["file_id"].duplicated().sum()

    duplicate_paths = (
        master["processed_audio_path"].duplicated().sum()
    )

    if duplicate_ids:
        print(
            f"ERROR — duplicate file IDs: {duplicate_ids}"
        )
        sys.exit(1)

    if duplicate_paths:
        print(
            f"ERROR — duplicate processed paths: "
            f"{duplicate_paths}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MASTER MANIFEST CREATED")
    print("=" * 70)

    print(f"Output : {OUTPUT_MANIFEST}")
    print(f"Rows   : {len(master)}")
    print(f"Columns: {len(master.columns)}")

    print("\nLanguages:")
    print(
        master["language_name"]
        .value_counts()
        .to_string()
    )

    print("\nSplits:")
    print(
        master["split"]
        .value_counts()
        .to_string()
    )

    print("\nLabels:")
    print(
        master["label"]
        .value_counts()
        .to_string()
    )

    print("\nAudio:")
    print(
        "Sample rate:",
        sorted(master["sample_rate"].unique())
    )

    print(
        "Channels:",
        sorted(master["channels"].unique())
    )

    print("\nSUCCESS — REAL master manifest ready.")


if __name__ == "__main__":
    main()