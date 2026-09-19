import pandas as pd
from pathlib import Path

manifest_path = Path("dataset_v1/metadata/real_audio_manifest.csv")

duration_files = {
    "HI": Path("dataset_v1/raw/real/hindi/clip_durations.tsv"),
    "MR": Path("dataset_v1/raw/real/marathi/clip_durations.tsv"),
}

manifest = pd.read_csv(manifest_path)

# Remove any old duration columns from previous attempts.
for col in ["duration_sec", "clip"]:
    if col in manifest.columns:
        manifest.drop(columns=[col], inplace=True)

duration_tables = []

for language, path in duration_files.items():

    df = pd.read_csv(path, sep="\t")

    df["language"] = language
    df["duration_sec"] = df["duration[ms]"] / 1000.0

    duration_tables.append(
        df[["clip", "language", "duration_sec"]]
    )

durations = pd.concat(
    duration_tables,
    ignore_index=True
)

# Merge duration using filename + language.
manifest = manifest.merge(
    durations,
    left_on=["original_filename", "language"],
    right_on=["clip", "language"],
    how="left",
    validate="many_to_one"
)

# Remove duplicate helper column.
manifest.drop(columns=["clip"], inplace=True)

# Quality classification.
manifest["quality_status"] = "usable"

manifest.loc[
    manifest["duration_sec"].isna(),
    "quality_status"
] = "missing_duration"

manifest.loc[
    manifest["duration_sec"] < 1.0,
    "quality_status"
] = "excluded_duration"

manifest.loc[
    manifest["duration_sec"] > 10.0,
    "quality_status"
] = "excluded_duration"

# Save updated manifest.
manifest.to_csv(
    manifest_path,
    index=False
)

print("Updated:", manifest_path)
print()

print("=== QUALITY STATUS ===")
print(
    manifest["quality_status"]
    .value_counts()
    .to_string()
)

print()
print("=== USABLE CLIPS BY LANGUAGE ===")

usable = manifest[
    manifest["quality_status"] == "usable"
]

print(
    usable.groupby("language_name")
    .size()
    .to_string()
)

print()
print("=== USABLE CLIPS BY LANGUAGE + SPLIT ===")

print(
    usable.groupby(
        ["language_name", "split"]
    )
    .size()
    .unstack(fill_value=0)
)

print()
print("=== DURATION STATISTICS ===")

print(
    usable["duration_sec"]
    .describe()
    .to_string()
)

