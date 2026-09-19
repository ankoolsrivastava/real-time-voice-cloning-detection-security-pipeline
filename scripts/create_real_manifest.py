import pandas as pd
from pathlib import Path

SEED = 42
MAX_CLIPS_PER_SPEAKER = 30

datasets = {
    "HI": {
        "metadata": Path("dataset_v1/raw/real/hindi/validated.tsv"),
        "audio_dir": Path("dataset_v1/raw/real/hindi/clips"),
    },
    "MR": {
        "metadata": Path("dataset_v1/raw/real/marathi/validated.tsv"),
        "audio_dir": Path("dataset_v1/raw/real/marathi/clips"),
    },
}

split_path = Path("dataset_v1/metadata/speaker_split.csv")
output_path = Path("dataset_v1/metadata/real_audio_manifest.csv")

speaker_split = pd.read_csv(split_path)

all_selected = []

for language, cfg in datasets.items():

    df = pd.read_csv(cfg["metadata"], sep="\t")

    # Language is known from the dataset we are processing.
    df["language"] = language

    # Create the same internal speaker ID used in speaker_inventory.csv
    df["speaker_id"] = language + "_" + df["client_id"].astype(str)

    # Attach speaker-level train/validation/test split
    df = df.merge(
        speaker_split,
        on=["speaker_id", "language"],
        how="inner",
        validate="many_to_one"
    )

    # Build absolute/relative source path
    df["source_path"] = df["path"].apply(
        lambda x: str(cfg["audio_dir"] / str(x))
    )

    # Verify audio exists
    df["file_exists"] = df["source_path"].apply(
        lambda x: Path(x).exists()
    )

    before = len(df)
    df = df[df["file_exists"]].copy()

    print(f"{language}: {before} metadata rows")
    print(f"{language}: {len(df)} rows with existing audio")

    selected_groups = []

    # Cap clips per speaker while preserving speaker diversity
    for speaker_id, group in df.groupby("speaker_id", sort=False):

        group = group.sample(
            frac=1.0,
            random_state=SEED
        )

        group = group.head(MAX_CLIPS_PER_SPEAKER)

        selected_groups.append(group)

    if selected_groups:
        selected = pd.concat(
            selected_groups,
            ignore_index=True
        )
    else:
        continue

    selected["label"] = "bonafide"
    selected["source"] = "CommonVoice"
    selected["condition"] = "natural"

    selected["language_name"] = selected["language"].map({
        "HI": "Hindi",
        "MR": "Marathi"
    })

    selected["generator"] = "none"
    selected["generator_version"] = "none"
    selected["duration_sec"] = None

    selected = selected.rename(
        columns={
            "path": "original_filename",
            "sentence": "transcript"
        }
    )

    keep_columns = [
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
        "duration_sec",
        "age",
        "gender",
        "accents",
        "client_id",
    ]

    selected = selected[keep_columns]

    all_selected.append(selected)

manifest = pd.concat(
    all_selected,
    ignore_index=True
)

# Give every selected clip an internal ID
manifest.insert(
    0,
    "file_id",
    [
        f"REAL_{i:06d}"
        for i in range(1, len(manifest) + 1)
    ]
)

manifest.to_csv(
    output_path,
    index=False
)

print()
print(f"Created: {output_path}")
print(f"Total selected clips: {len(manifest)}")
print()

print("=== CLIP COUNTS ===")
print(
    manifest.groupby(
        ["language_name", "split"]
    )
    .size()
    .unstack(fill_value=0)
)

print()
print("=== SPEAKER COUNTS ===")
print(
    manifest.groupby(
        ["language_name", "split"]
    )["speaker_id"]
    .nunique()
    .unstack(fill_value=0)
)

print()
print("=== CLIPS PER SPEAKER ===")
print(
    manifest.groupby(
        ["language_name", "split", "speaker_id"]
    )
    .size()
    .groupby(level=[0, 1])
    .agg(["min", "median", "mean", "max"])
)

